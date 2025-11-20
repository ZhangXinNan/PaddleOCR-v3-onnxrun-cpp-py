import os
import argparse
import cv2
import numpy as np
import json
from text_det import TextDetector
from text_angle_cls import TextClassifier
from text_rec import TextRecognizer
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='images/1.jpg', help="image path")
    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    detect_model = TextDetector()
    angle_model = TextClassifier()
    rec_model = TextRecognizer()

    srcimg = cv2.imread(args.input)
    img_show = srcimg.copy()
    # srcimg = cv2.rotate(srcimg, 1)
    box_list = detect_model.detect(srcimg)
    text = ''
    if len(box_list) > 0:
        result = []
        for j, point in enumerate(box_list):
            point = detect_model.order_points_clockwise(point)
            textimg = detect_model.get_rotate_crop_image(srcimg, point.astype(np.float32))
            angle = angle_model.predict(textimg)
            if angle=='180':
                textimg = cv2.rotate(textimg, 1)
            text = rec_model.predict_text(textimg)

            point = point.astype(int)
            cv2.polylines(img_show, [point], True, (0, 0, 255), thickness=2)
            # for i in range(4):
            #     cv2.circle(img_show, tuple(point[i, :]), 2, (0, 255, 0), thickness=-1)
            print(j, point.tolist(), text)
            # fo.write(f"{','.join(map(str, point.flatten()))}\t{text}\n")
            result.append({"box": point.tolist(), "text": text})

        fo = open(os.path.join(args.out_dir, os.path.basename(args.input) + ".result.txt"), 'w', encoding='utf-8')
        json.dump(result, fo, indent=4, ensure_ascii=False)
        fo.close()
    save_path = os.path.join(args.out_dir, os.path.basename(args.input) + ".show.jpg")
    cv2.imwrite(save_path, img_show)
    '''
    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('result.jpg', srcimg)
    '''



    