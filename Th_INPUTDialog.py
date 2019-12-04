"""
特徴抽出(sobel,laplacianなど)後に画像を二値化する際、しきい値に小数点を指定することができるCLI

作った理由
画像をシークバーをいじって二値化したかったが、cv2.createTrackbarではシークバーにより小数点を指定することができないのでINPUT関数よりしきい値を指定して二値化できる

使用方法
実行後二値化のしきい値を入力
->画像が出力される
->画像上でEscキーを押す
->continue?と聞かれるのでyで繰り返し処理を行う、それ以外中なら終わり

特徴抽出法を適宜変える


"""
import numpy as np
import cv2

#laplacianフィルタVer
def lap_gs(blur_img):
    kekka = cv2.Laplacian(blur_img,cv2.CV_64F)

    return kekka

# Sobelフィルタを用いて特徴量検出　入力np.array型の画素値
def Sobel_RGB(Image):
	sobel_edge_ = np.empty((Image.shape[0], Image.shape[1], 3), dtype=np.uint8)
	for i in range(0,3):
		# x.y方向の特徴量を得る
		sobel_image_x = cv2.Sobel(Image[:,:,i],cv2.CV_32F,1,0)
		sobel_image_y = cv2.Sobel(Image[:,:,i],cv2.CV_32F,0,1)
		# それぞれを8ビット変換
		abs_sobel_x = cv2.convertScaleAbs(sobel_image_x)
		abs_sobel_y = cv2.convertScaleAbs(sobel_image_y)
		# X、Yの重みを半々にして一つの画素値を取得
		sobel_edge_temp = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
		sobel_edge_[:,:,i] = sobel_edge_temp
	
	print(sobel_edge_.shape)
	# sobel_edge_の中に格納されているRGBのSobel値から最大のものを出力
	sobel_edge = np.max(sobel_edge_, axis=2)
	return sobel_edge

# しきい値(float)によって二値化する
def Threshold_image(edge_image,Th):
	ret, img_thresh = cv2.threshold(edge_image, Th, 255, cv2.THRESH_BINARY)
	return img_thresh

if __name__ == '__main__':
    img = cv2.imread('/home/atsushi/デスクトップ/reseaching/image/big_depth/sitaba/dep10008_resize.tif', cv2.IMREAD_GRAYSCALE)
    # 特徴抽出法　変えるならここ
    edge_img = lap_gs(img)
    # ##########
    cv2.namedWindow("edge")
    finish="y"

    while(finish == "y"):
        Th_char = input("input Threshold: ")
        Th = float(Th_char)
        Th_img = Threshold_image(edge_img,Th)
        print("press Esc on edge window")
        cv2.imshow("edge",Th_img)
        cv2.waitKey(0)
        finish = input("continue? y/n")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()