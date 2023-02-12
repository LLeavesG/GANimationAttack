import json
import cv2
from face import detect, img2face, cascade, face2img
 
post_data = {}
rects_keys = ['x1', 'y1', 'x2', 'y2']

'''
def get_face(imgname):
    retval = {}
    try:
        img = cv2.imread('../upload/' + imgname)
        if img is not None:
            rects = detect(img, cascade)
            img2face(img, rects, imgname)

            if len(rects) != 0:
                retval['code'] = 200
                retval['msg'] = 'Success'

                for i in range(4):
                    retval[rects_keys[i]] = int(rects[0][i])
            else:
                retval['code'] = 400
                retval['msg'] = 'Can not detect face'
        else:
            retval['code'] = 401
            retval['msg'] = 'File not exist' 
    except:
        retval['code'] = 402
        retval['msg'] = 'Unknow Error' 
    
    return json.dumps(retval)
'''
args = {}
rects = []
args['x1'] = 181
args['y1'] = 227
args['x2'] = 842
args['y2'] = 888
args['image_name'] = 'eade4f553e4d2f6fba310fdcf3052fc0.jpg'
for i in range(4):
    rects.append(args[rects_keys[i]])

face2img(args['image_name'], rects)