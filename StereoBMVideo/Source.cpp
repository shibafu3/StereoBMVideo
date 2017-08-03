#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib")

#pragma comment(lib, "C:\\Program Files\\Anaconda3\\libs\\python35.lib")
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

int main(){
	Mat Q;
	Mat mapx_l, mapy_l;
	Mat mapx_r, mapy_r;
	//xmlファイルから必要なマップデータを読み込み
	FileStorage cvfsr("C:\\Users\\0133752\\Desktop\\StereoCalibrate.xml", FileStorage::READ);
	FileNode node(cvfsr.fs, NULL);
	read(node["mapx_l"], mapx_l);
	read(node["mapy_l"], mapy_l);
	read(node["mapx_r"], mapx_r);
	read(node["mapy_r"], mapy_r);
	read(node["Q"], Q);


	VideoCapture capture_l(1), capture_r(2);

	cv::Mat image_l;
	cv::Mat image_r;

	//ステレオマッチング用の各種パラメータ
	int window_size = 3;
	int minDisparity = 0;
	int numDisparities = 80;
	int blockSize = 11;
	int P1 = 8 * 3 * window_size * window_size;
	int P2 = 32 * 3 * window_size * window_size;
	int disp12MaxDiff = 1;
	int preFilterCap = 0;
	int uniquenessRatio = 10;
	int speckleWindowSize = 100;
	int speckleRange = 1;

	//minDisparity------------ - 最小視差値、一般には０で良い
	//numDisparities---------- - 最大視差値と最小視差値の差、１６の倍数にする
	//SADWindowSize----------Sum of Absolute Differencesを計算するウィンドウのサイズ、３～１１の奇数を推奨
	//P1 = 0 ---------------- - 値の決め方が示されているが、デフォルトのままで良かった
	//P2 = 0 ---------------- - 同上
	//disp12MaxDiff = 0 --------左右の視差の許容最大値、デフォルトはチェックしない、デフォルトで良かった
	//preFilterCap = 0 ----------事前にフィルタで大きな視差をクリップする、デフォルトで良かった
	//uniquenessRatio = 0 ------目的関数値の次点との差の％比率、０は比較しない意、デフォルトで十分であった
	//speckleWindowSize = 0 ----小さい斑点やノイズを消すフィルタのサイズ、０のデフォルトは使用しない意、これは使って非常に効果があった。
	//speckleRange = 0 --------上記フィルタを使用するときの、視差の最大値、１～２が推奨で、１６倍される、１が良かった
	//fullDP = false ------------フルスペックのダイナミックプログラミングを行わないのがデフォルト、trueにしても大差はなかった


	//ステレオBMのインスタンス化
	//StereoBMのcreateメソッドでポインタを取得
	//その際に各種パラメータを引数に入力
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(
		numDisparities,
		blockSize);
		//numDisparities = 0 -- - int、0はデフォルト値ではなく、デフォルトの意。求める視差の最大値を16の倍数で指定する。デフォルト値は48以上と推定される
		//blockSize = 21 ------int、Sum of Absolute Differencesを計算するウィンドウのサイズ、奇数であること、資料には3～11とあるが、5以上でないとエラーになる、大きいと広範囲で平均化され、小さいと細かくなるが、ノイズに弱くなる。

	while (1){
		//左右画像読み込み
		capture_l >> image_l;
		capture_r >> image_r;

		//ステレオカメラ用の補正マップを用いて入力画像を補正
		remap(image_l, image_l, mapx_l, mapy_l, INTER_LINEAR);
		remap(image_r, image_r, mapx_r, mapy_r, INTER_LINEAR);

		cvtColor(image_l, image_l, CV_BGR2GRAY);
		cvtColor(image_r, image_r, CV_BGR2GRAY);

		imshow("image_l", image_l);
		imshow("image_r", image_r);

		//左右画像から深度マップを作成
		cv::Mat disparity;    //((cv::MatSize)leftImg.size, CV_16S);
		sbm->compute(image_l, image_r, disparity);

		Mat _3dImage;
		reprojectImageTo3D(disparity, _3dImage, Q);


		//深度マップを視覚的に分かるようにピクセル値を変換
		cv::Mat disparity_map;
		double min, max;
		//深度マップの最小値と最大値を求める
		cv::minMaxLoc(disparity, &min, &max);
		//CV_8UC1に変換、最大レンジを0～255にする
		disparity.convertTo(disparity_map, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));
		cv::imshow("result", disparity_map);

		if (cv::waitKey(10) == 27){
			break;
		}
	}

	return 0;
}