#ifdef _DEBUG
//Debug���[�h�̏ꍇ
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Release���[�h�̏ꍇ
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
	//xml�t�@�C������K�v�ȃ}�b�v�f�[�^��ǂݍ���
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

	//�X�e���I�}�b�`���O�p�̊e��p�����[�^
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

	//minDisparity------------ - �ŏ������l�A��ʂɂ͂O�ŗǂ�
	//numDisparities---------- - �ő压���l�ƍŏ������l�̍��A�P�U�̔{���ɂ���
	//SADWindowSize----------Sum of Absolute Differences���v�Z����E�B���h�E�̃T�C�Y�A�R�`�P�P�̊�𐄏�
	//P1 = 0 ---------------- - �l�̌��ߕ���������Ă��邪�A�f�t�H���g�̂܂܂ŗǂ�����
	//P2 = 0 ---------------- - ����
	//disp12MaxDiff = 0 --------���E�̎����̋��e�ő�l�A�f�t�H���g�̓`�F�b�N���Ȃ��A�f�t�H���g�ŗǂ�����
	//preFilterCap = 0 ----------���O�Ƀt�B���^�ő傫�Ȏ������N���b�v����A�f�t�H���g�ŗǂ�����
	//uniquenessRatio = 0 ------�ړI�֐��l�̎��_�Ƃ̍��́��䗦�A�O�͔�r���Ȃ��ӁA�f�t�H���g�ŏ\���ł�����
	//speckleWindowSize = 0 ----���������_��m�C�Y�������t�B���^�̃T�C�Y�A�O�̃f�t�H���g�͎g�p���Ȃ��ӁA����͎g���Ĕ��Ɍ��ʂ��������B
	//speckleRange = 0 --------��L�t�B���^���g�p����Ƃ��́A�����̍ő�l�A�P�`�Q�������ŁA�P�U�{�����A�P���ǂ�����
	//fullDP = false ------------�t���X�y�b�N�̃_�C�i�~�b�N�v���O���~���O���s��Ȃ��̂��f�t�H���g�Atrue�ɂ��Ă��卷�͂Ȃ�����


	//�X�e���IBM�̃C���X�^���X��
	//StereoBM��create���\�b�h�Ń|�C���^���擾
	//���̍ۂɊe��p�����[�^�������ɓ���
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(
		numDisparities,
		blockSize);
		//numDisparities = 0 -- - int�A0�̓f�t�H���g�l�ł͂Ȃ��A�f�t�H���g�̈ӁB���߂鎋���̍ő�l��16�̔{���Ŏw�肷��B�f�t�H���g�l��48�ȏ�Ɛ��肳���
		//blockSize = 21 ------int�ASum of Absolute Differences���v�Z����E�B���h�E�̃T�C�Y�A��ł��邱�ƁA�����ɂ�3�`11�Ƃ��邪�A5�ȏ�łȂ��ƃG���[�ɂȂ�A�傫���ƍL�͈͂ŕ��ω�����A�������ƍׂ����Ȃ邪�A�m�C�Y�Ɏキ�Ȃ�B

	while (1){
		//���E�摜�ǂݍ���
		capture_l >> image_l;
		capture_r >> image_r;

		//�X�e���I�J�����p�̕␳�}�b�v��p���ē��͉摜��␳
		remap(image_l, image_l, mapx_l, mapy_l, INTER_LINEAR);
		remap(image_r, image_r, mapx_r, mapy_r, INTER_LINEAR);

		cvtColor(image_l, image_l, CV_BGR2GRAY);
		cvtColor(image_r, image_r, CV_BGR2GRAY);

		imshow("image_l", image_l);
		imshow("image_r", image_r);

		//���E�摜����[�x�}�b�v���쐬
		cv::Mat disparity;    //((cv::MatSize)leftImg.size, CV_16S);
		sbm->compute(image_l, image_r, disparity);

		Mat _3dImage;
		reprojectImageTo3D(disparity, _3dImage, Q);


		//�[�x�}�b�v�����o�I�ɕ�����悤�Ƀs�N�Z���l��ϊ�
		cv::Mat disparity_map;
		double min, max;
		//�[�x�}�b�v�̍ŏ��l�ƍő�l�����߂�
		cv::minMaxLoc(disparity, &min, &max);
		//CV_8UC1�ɕϊ��A�ő僌���W��0�`255�ɂ���
		disparity.convertTo(disparity_map, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));
		cv::imshow("result", disparity_map);

		if (cv::waitKey(10) == 27){
			break;
		}
	}

	return 0;
}