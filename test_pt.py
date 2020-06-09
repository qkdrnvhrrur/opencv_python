import cv2;
import sys;
import os;
import pytesseract as pt;
import numpy as np;

model="./front_trained/frozen_inference_graph.pb";
config="./front_trained/graph.pbtxt";

str_path_dir="data\\test\strings";
img_path_dir="data\images";

str_file_list=os.listdir(str_path_dir);
img_file_list=os.listdir(img_path_dir);

net=cv2.dnn.readNet(model, config);

def getPlate(file):
    img=cv2.imread(file, cv2.IMREAD_COLOR);
    height, width, channel=img.shape;

    blob=cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123));
    net.setInput(blob);
    res=net.forward();

    detect=res.reshape(res.shape[2], res.shape[3]);

    for d in detect:
        confidence=d[2];
        if(confidence<0.5):
            continue;

        x1=int(round(d[3]*width));
        y1=int(round(d[4]*height));
        x2=int(round(d[5]*width));
        y2=int(round(d[6]*height));

        return img[y1:y2, x1:x2];

def getErosion(lp):
    tmp=cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY);
    ret, binImg=cv2.threshold(tmp, 0, 255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU);

    kernel_size_row = 3;
    kernel_size_col = 3;
    kernel = np.ones((3, 3), np.uint8);
    
    return cv2.erode(binImg, kernel, iterations=1);

'''license_plate=getPlate("./116.jpg");
cv2.imshow("lp", license_plate);
cv2.waitKey(0);

erosion=getErosion(license_plate);
cv2.imshow("erosion", erosion);
cv2.waitKey(0);

cv2.destroyAllWindows();

print(pt.image_to_string(erosion, lang="kor"));'''

for f in img_file_list:
    license_plate=getPlate(os.path.join(img_path_dir, f));

    if license_plate is not None:
        erosion=getErosion(license_plate);

        print(f, pt.image_to_string(erosion, lang="kor"));
