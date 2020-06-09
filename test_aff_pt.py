import cv2;
import numpy as np;
import sys;
import os;
import pytesseract as pt;

model="./front_trained/frozen_inference_graph.pb";
config="./front_trained/graph.pbtxt";

def main():
    net=cv2.dnn.readNet(model, config);
    if(net.empty()):
        sys.exit("net error");

    src=cv2.imread('./data/images/image11.jpg', cv2.IMREAD_COLOR);
    #cv2.imshow('original img', src);
    #cv2.waitKey(0);
    img = cv2.resize(src, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA);
    #cv2.imshow('resized img', img);
    #cv2.waitKey(0);
    height, width, channel=img.shape;

    print("detecting license plate...");
    blob=cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123));
    net.setInput(blob);
    res=net.forward();

    detect=res.reshape(res.shape[2], res.shape[3]);

    for d in detect:
        confidence=d[2];
        print(confidence);
        if(confidence<0.5):
            continue;

        x1=int(round(d[3]*width));
        y1=int(round(d[4]*height));
        x2=int(round(d[5]*width));
        y2=int(round(d[6]*height));

        license_plate=img[y1:y2, x1:x2];
        print(x2-x1, y2-y1);
        cv2.imshow('original lp', license_plate);
        cv2.waitKey(0);
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2);

    print("found license plate!!!");
    print("-----------------------------------------------------");
    print("retrieving license plate number from license plate...");
    
    w=300; h=100;
    srcQ=np.float32([(x1+15, y1), (x2-20, y2-50), (x2-20, y2), (x1+15, y1+50)]);
    dstQ=np.float32([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]);
    pers=cv2.getPerspectiveTransform(srcQ, dstQ);
    aff=cv2.warpPerspective(img, pers, (w, h));
    
    aff_gray=cv2.cvtColor(aff, cv2.COLOR_BGR2GRAY);
    ret, binAff=cv2.threshold(aff_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU);
    cv2.imshow('binAff', binAff);
    
    kernel = np.ones((3, 3), np.uint8);
    #dil=cv2.dilate(binAff, kernel, iterations=1);
    close=cv2.morphologyEx(binAff, cv2.MORPH_CLOSE, kernel, iterations=1);
    cv2.imshow("close", close);
    
    final=pt.image_to_string(close, lang="kor");
    print(final);
    img=cv2.putText(img, final, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('final img', img);
    cv2.imshow("detected lp", license_plate);
    cv2.imshow("aff", aff);

    cv2.waitKey(0);
    cv2.destroyAllWindows();

main();