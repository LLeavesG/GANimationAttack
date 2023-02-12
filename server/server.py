from crypt import methods
from curses.ascii import isalpha, isdigit
import json
from mimetypes import suffix_map
from flask import Flask, request
from gevent import monkey
from gevent.pywsgi import WSGIServer
import cv2
from face import detect, face2img, img2face, cascade, rects_keys
from solver import Solver
import base64

monkey.patch_all()
solver = Solver()

app = Flask(__name__)

def name_check(name):
    # safe check for image name
    if len(name) == 36:
        if name[-4:] == '.jpg' and './' not in name and '.\\' not in name:
            for each in name[:-4]:
                if isdigit(each) or isalpha(each):
                    return True
    return False

def args_check(args):
    # safe check for protect args
    if type(args['protect_class_conditional']) == bool:
        if type(args['protect_expression_seq']) == int:
            if args['protect_expression_seq'] >= 0 and args['protect_expression_seq'] < 28:
                if name_check(args['image_name']):
                    return True

    return False



@app.route('/getface/<imgname>',methods=['GET'])
def get_face(imgname):
    retval = {}
    try:
        img = cv2.imread('../upload/' + imgname)
        if img is not None:
            rects = detect(img, cascade)
            img2face(img, rects, imgname)

            if len(rects) != 0:
                retval['code'] = 100
                retval['msg'] = 'Success'
                for i in range(4):
                    retval[rects_keys[i]] = int(rects[0][i])
            else:
                retval['code'] = 101
                retval['msg'] = 'Can not detect face'
        else:
            retval['code'] = 102
            retval['msg'] = 'File not exist' 
    except:
        retval['code'] = 103
        retval['msg'] = 'Unknow error' 
    
    return json.dumps(retval)


@app.route('/protect/', methods=['GET', 'POST'])
def protect():
    retval = {}
    try:
        if request.method == 'POST':
            args = request.get_data()
            args = json.loads(args)
            if args_check(args):
                solver.protect(args)
                face2img(args)
                retval['code'] = 200
                retval['msg'] = 'Success'
            else:
                retval['code'] = 201
                retval['msg'] = 'Input args error'
        else:
            retval['code'] = 202
            retval['msg'] = 'Only method POST can work'
    except:
        retval['code'] = 203
        retval['msg'] = 'Unknow error'
        
    return json.dumps(retval)
    

@app.route('/protected/',methods=['POST'])
def get_protected():
    retval = {}
    try:
        if request.method == 'POST':
            args = request.get_data()
            args = json.loads(args)

            if args_check(args):
                retval['code'] = 300
                retval['msg'] = 'Success'
                imgname = args['image_name'][:-4]
                

                if args['protect_class_conditional'] == True:
                    suffix = 'all.jpg'
                else:
                    suffix = str(args['protect_expression_seq'] + 1) + '.jpg'

                file = open("../image_protect/" + imgname + '_' + suffix, "rb")
                res = file.read()
                b64code = base64.b64encode(res).decode('ascii')
                retval['data'] = b64code
            else:
                retval['code'] = 301
                retval['msg'] = 'Input args error'
        else:
            retval['code'] = 302
            retval['msg'] = 'Only method POST can work'
    except:
        retval['code'] = 303
        retval['msg'] = 'Unknow error'

    return json.dumps(retval)


@app.route('/gen_protected/',methods=['POST'])
def get_gen_protected():
    retval = {}
    try:
        if request.method == 'POST':
            args = request.get_data()
            args = json.loads(args)
            if args_check(args):
                retval['code'] = 400
                retval['msg'] = 'Success'
            else:
                retval['code'] = 401
                retval['msg'] = 'Input args error'
            
            imgname = args['image_name'][:-4]
            if args['protect_class_conditional'] == True:
                suffix = 'all.jpg';
            else:
                suffix = str(args['protect_expression_seq'] + 1) + '.jpg'

            file = open("../image_protect_gen/" + imgname + '_' + suffix, "rb")
            res = file.read()
            b64code = base64.b64encode(res).decode('ascii')
            retval['data'] = b64code

        else:
            retval['code'] = 402
            retval['msg'] = 'Only method POST can work'
    except:
        retval['code'] = 403
        retval['msg'] = 'Unknow error'
        
    return json.dumps(retval)

@app.route('/gen/',methods=['POST'])
def get_gen():
    retval = {}
    try:
        if request.method == 'POST':
            args = request.get_data()
            args = json.loads(args)
            if args_check(args):
                retval['code'] = 500
                retval['msg'] = 'Success'
            else:
                retval['code'] = 501
                retval['msg'] = 'Input args error'
            
            imgname = args['image_name'][:-4]
            if args['protect_class_conditional'] == True:
                suffix = 'all.jpg';
            else:
                suffix = str(args['protect_expression_seq'] + 1) + '.jpg'

            file = open("../image_gen/" + imgname + '_' + suffix, "rb")
            res = file.read()
            b64code = base64.b64encode(res).decode('ascii')
            retval['data'] = b64code

        else:
            retval['code'] = 502
            retval['msg'] = 'Only method POST can work'
    except:
        retval['code'] = 503
        retval['msg'] = 'Unknow error'
        
    return json.dumps(retval)


@app.route("/")
def index():
    return 'Hello'

if __name__ == "__main__":

    http_server = WSGIServer(('0.0.0.0', 80), app)
    http_server.serve_forever()
    #app.run(host='0.0.0.0', port=80)