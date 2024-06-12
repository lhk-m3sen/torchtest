#include "main.hpp"

using namespace std;
using namespace cv;

bool isAiModelAvailable;
torch::jit::script::Module module;

int init(string model, bool cudaOk)
{
    isAiModelAvailable = false;
    if (!filesystem::exists(model))
    {
        cerr << "error: model not found" << endl;
        return -1;
    }
    try
    {
        module = torch::jit::load(model);
        if (cudaOk)
            module.to(torch::kCUDA);
        isAiModelAvailable = true;
    }
    catch (const c10::Error &e)
    {
        cerr << "error loading the model: " << e.what() << endl;
        return -1;
    }
    return 0;
}

int FilterTest(string rgbPath, vector<string> &labels, bool cudaOk)
{
    if (!isAiModelAvailable)
        return -1;

    Mat matColor = imread(rgbPath, cv::IMREAD_COLOR);
    cvtColor(matColor, matColor, cv::COLOR_BGR2RGB);
    resize(matColor, matColor, Size(AIFILTER_IMG, AIFILTER_IMG));
    matColor.convertTo(matColor, CV_32FC3, 1.0f / 255.0f);

    // mat to tensor
    torch::Tensor inputTensor;
    if (cudaOk)
    {
        inputTensor =
            torch::from_blob(matColor.data, {1, AIFILTER_IMG, AIFILTER_IMG, 3}, torch::kFloat32).to(torch::kCUDA);
    }
    else
    {
        inputTensor = torch::from_blob(matColor.data, {1, AIFILTER_IMG, AIFILTER_IMG, 3});
    }
    inputTensor = inputTensor.permute({0, 3, 1, 2});

    // model forward
    torch::Tensor outTensor = module.forward({inputTensor}).toTensor();
    if (cudaOk)
        outTensor = outTensor.to(torch::kCPU);
    cout << "outTensor: " << outTensor.slice(1, 0, 8) << endl;
    auto sortResult = outTensor.sort(-1, true);
    cout << "sortResult[0]: " << get<0>(sortResult) << "\t sortResult[1]: " << get<1>(sortResult) << endl;
    auto softmaxResult = get<0>(sortResult)[0].softmax(0);
    cout << "softmaxResult: " << softmaxResult.slice(0, 0, 8) << endl;
    auto index = get<1>(sortResult)[0];

    auto idx = index[0].item<int>();
    cout << "index:  " << idx << endl;
    cout << "Label: " << labels[idx] << endl;
    cout << "probability:  " << softmaxResult[0].item<float>() * 100.0f << "%" << endl;

    return 0;
}

int LoadLabel(string labelPath, vector<string> &labels)
{
    ifstream labelFile(labelPath);
    if (!labelFile)
        return -1;

    string labelLine;
    while (getline(labelFile, labelLine))
        labels.push_back(labelLine);

    return 0;
}

int main(int, char **)
{
    int result_val;
    vector<string> labels;

    string modelPath = "../../model.pt";
    string rgbPath = "../../color.jpg";
    string labelPath = "../../label.txt";

    bool isCuda = torch::cuda::is_available() ? true : false;
    cout << "cuda available: " << isCuda << endl;

    LoadLabel(labelPath, labels);

    init(modelPath, isCuda);
    FilterTest(rgbPath, labels, isCuda);
}
