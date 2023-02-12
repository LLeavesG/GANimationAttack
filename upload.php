<?php
class Upload{

    function __construct() {
        $this->filename = '';
        $this->filedir = '';
   }

    public function upload_init(){
        // 上传前置操作
    }

    public function uploadimg()
    {
        $this->upload_init();
        if(!empty($_FILES['file'])) {
            
            var_dump($_FILES['file']);

            $filename = md5(date("D-M-Y-H-i-s").'salt_lleaves') . '.jpg';
            $filedir = 'upload/'.$filename;
            move_uploaded_file($_FILES['file']['tmp_name'], $_SERVER['DOCUMENT_ROOT'].'/'.$filedir);
            var_dump('上传成功, 保存于'.$filedir);
            $this->after_upload();
        }
    }

    public function after_upload(){
        // 上传完成后操作
    }
}

$upload_obj = new Upload;
$upload_obj->uploadimg();


