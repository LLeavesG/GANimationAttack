import cv2


# result iamge size
size_m = 384
size_n = 384
cascade = cv2.CascadeClassifier('../utils/haarcascade_frontalface_alt2.xml')

rects_keys = ['x1', 'y1', 'x2', 'y2']
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def img2face(dst, rects, out):
    for x1, y1, x2, y2 in rects:
        roi = dst[y1+10:y2+20, x1+10:x2]
        img_roi = roi
        re_roi = cv2.resize(img_roi, (size_m, size_n),
                            interpolation=cv2.INTER_AREA)

        cv2.imwrite('../image_src/' + out, re_roi)
    return



def face2img(args):

    rects = []

    for i in range(4):
        rects.append(args[rects_keys[i]])

    org_img = cv2.imread('../upload/' + args['image_name'])
    if args['protect_class_conditional'] == True:
        suffix = '_all'
    else:
        suffix = '_' + str(args['protect_expression_seq'] + 1)

    face_path = '../image_protect/' + args['image_name'][:-4] + suffix + '.jpg'
    
    x1 = rects[0]
    y1 = rects[1]
    x2 = rects[2]
    y2 = rects[3]

    face = cv2.imread(face_path)
    face = cv2.resize(face, (x2 - x1 - 10, y2 - y1 + 10), interpolation=cv2.INTER_CUBIC)
    org_img[y1+10:y2+20, x1+10:x2] = face

    cv2.imwrite(face_path, org_img)


    