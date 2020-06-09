import json;
import cv2;
import boto3;
import base64;
import numpy as np;
import email;

def id2class(num):
    if num is 10:
        return 'Ga';
    elif num is 11:
        return 'Na';
    elif num is 12:
        return 'Da';
    elif num is 13:
        return 'Ra';
    elif num is 14:
        return 'Ma';
    elif num is 15:
        return 'Geo';
    elif num is 16:
        return 'Neo';
    elif num is 17:
        return 'Deo';
    elif num is 18:
        return 'Reo';
    elif num is 19:
        return 'Meo';
    elif num is 20:
        return 'Beo';
    elif num is 21:
        return 'Seo';
    elif num is 22:
        return 'Eo';
    elif num is 23:
        return 'Jeo';
    elif num is 24:
        return 'Go';
    elif num is 25:
        return 'No';
    elif num is 26:
        return 'Do';
    elif num is 27:
        return 'Ro';
    elif num is 28:
        return 'Mo';
    elif num is 29:
        return 'Bo';
    elif num is 30:
        return 'So';
    elif num is 31:
        return 'O';
    elif num is 32:
        return 'Jo';
    elif num is 33:
        return 'Gu';
    elif num is 34:
        return 'Nu';
    elif num is 35:
        return 'Du';
    elif num is 36:
        return 'Ru';
    elif num is 37:
        return 'Mu';
    elif num is 38:
        return 'Bu';
    elif num is 39:
        return 'Su';
    elif num is 40:
        return 'U';
    elif num is 41:
        return 'Ju';
    elif num is 42:
        return 'Heo';
    elif num is 43:
        return 'Ha';
    elif num is 44:
        return 'Ho';
    else:
        return str(num);
        
s3=boto3.client("s3");
    
s3.download_file('smart-parking-lot', 'frozen_inference_graph.pb', '/tmp/frozen_inference_graph.pb');
model='/tmp/frozen_inference_graph.pb';

s3.download_file('smart-parking-lot', 'graph.pbtxt', '/tmp/graph.pbtxt');
config='/tmp/graph.pbtxt';

s3.download_file('smart-parking-lot', 'digits_svm_model.yml', '/tmp/digits_svm_model.yml');
svm_model='/tmp/digits_svm_model.yml';

winSize=(30, 50);
blockSize=(10, 10);
blockStride=(5, 5);
cellSize=(5, 5);
nbins=9;

def lambda_handler(event, context):
    body=base64.b64decode(event['body']);
    boundary=body[0:52];
    content_start=body.find(b'\r\n\r\n')+4;
    file_content=body[content_start:body.find(boundary, 1)];
    
    np_array=np.fromstring(file_content, np.uint8);
    src=cv2.imdecode(np_array, cv2.IMREAD_COLOR);
    img = cv2.resize(src, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA);
    height, width, channel=img.shape;
    
    net=cv2.dnn.readNet('/tmp/frozen_inference_graph.pb', '/tmp/graph.pbtxt');
    if(net.empty()):
        sys.exit("net error");
    
    blob=cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123));
    net.setInput(blob);
    res=net.forward();
    
    detect=res.reshape(res.shape[2], res.shape[3]);
    for d in detect:
        confidence=d[2];
        if(confidence<0.1):
            continue;

        x1=int(round(d[3]*width));
        y1=int(round(d[4]*height));
        x2=int(round(d[5]*width));
        y2=int(round(d[6]*height));

        license_plate=img[y1:y2, x1:x2];
        
    svm=cv2.ml.SVM_load(svm_model);
    hog=cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
    
    w=300; h=150;
    srcQ=np.float32([(x1+10, y1), (x2-10, y2-50), (x2-10, y2), (x1+10, y1+50)]);
    dstQ=np.float32([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]);
    pers=cv2.getPerspectiveTransform(srcQ, dstQ);
    aff=cv2.warpPerspective(img, pers, (w, h));
    
    aff_gray=cv2.cvtColor(aff, cv2.COLOR_BGR2GRAY);
    ret, binAff=cv2.threshold(aff_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU);
    
    kernel = np.ones((3, 3), np.uint8);
    close=cv2.morphologyEx(binAff, cv2.MORPH_CLOSE, kernel, iterations=1);
    
    cnt, labels, stats, centroids=cv2.connectedComponentsWithStats(close);
    y_sorted=stats[np.argsort(stats[:, 1])];
    x_sorted=y_sorted[np.argsort(y_sorted[:, 0])];
    
    nums=[];
    prev_x=99; prev_y=99;
    prev_w=99; prev_h=99;
    n=0;
    
    for s in x_sorted:
        if s[2]>10 and s[2]<60 and s[3]>50 and s[3]<110 and s[4]>300 and s[4]<2000 and abs((s[0]+s[2]/2)-(prev_x+prev_w/2))>20 :
            roi=binAff[s[1]:s[1]+s[3], s[0]:s[0]+s[2]];
            cons=cv2.copyMakeBorder(roi, 3, 3, 3, 3, cv2.BORDER_CONSTANT, (0, 0, 0));
            
            dst=cv2.resize(cons, dsize=winSize, interpolation=cv2.INTER_CUBIC);
            descriptor=hog.compute(dst);
            testData=descriptor.reshape(-1, 1620);
            retval, num=svm.predict(testData);
            nums.append(id2class(int(num)));
            n+=1;
            
        if n==2 or n==3:
            if s[2]>10 and s[2]<60 and s[3]>20 and s[3]<55 and s[4]>150 and s[4]<1000 and abs((s[0]+s[2]/2)-(prev_x+prev_w/2))<10 and abs((s[1]+s[3]/2)-(prev_y+prev_h/2))>30 :
                if prev_y>s[1]:
                    roi=binAff[s[1]:s[1]+prev_h+s[3], prev_x:prev_x+s[2]];
                else:
                    roi=binAff[prev_y:prev_y+prev_h+s[3], prev_x:prev_x+s[2]];
            
                cons=cv2.copyMakeBorder(roi, 3, 3, 3, 3, cv2.BORDER_CONSTANT, (0, 0, 0));
                dst=cv2.resize(cons, dsize=winSize, interpolation=cv2.INTER_CUBIC);
                descriptor=hog.compute(dst);
                testData=descriptor.reshape(-1, 1620);
                retval, num=svm.predict(testData);
                nums.append(id2class(int(num)));
            
            elif s[2]>5 and s[2]<30 and s[3]>60 and s[3]<110 and s[4]>150 and s[4]<1000 and abs((s[0]+s[2]/2)-(prev_x+prev_w/2))<20 and abs((s[1]+s[3]/2)-(prev_y+prev_h/2))<10 :
                roi=binAff[s[1]:s[1]+s[3], prev_x:prev_x+prev_w+s[2]];
            
                cons=cv2.copyMakeBorder(roi, 3, 3, 3, 3, cv2.BORDER_CONSTANT, (0, 0, 0));
                dst=cv2.resize(cons, dsize=winSize, interpolation=cv2.INTER_CUBIC);
                descriptor=hog.compute(dst);
                testData=descriptor.reshape(-1, 1620);
                retval, num=svm.predict(testData);
                nums.pop();
                nums.append(id2class(int(num)));
        
        if s[2]>5 and s[3]>20 and s[4]>150 and s[4]<2000:
            prev_x=s[0]; prev_y=s[1]; prev_w=s[2]; prev_h=s[3];
            
    final=''.join(nums);
    
    return {
        'statusCode': 200,
        "body": json.dumps(final)
    }