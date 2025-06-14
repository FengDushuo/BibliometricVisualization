
$(document).ready(function() {
    var task_id = WebUploader.Base.guid();          //产生task_id
    var uploader = WebUploader.create({             //创建上传控件
        swf: '../static/webuploader/Uploader.swf',  //swf位置，这个可能与flash有关
        server: '/upload_file',                   //接收每一个分片的服务器地址
        pick: '#picker',                          //填上传按钮的id选择器值
        auto: true,                               //选择文件后，是否自动上传
        chunked: true,                            //是否分片
        chunkSize: 20 * 1024 * 1024,              //每个分片的大小，这里为20M
        chunkRetry: 3,                            //某分片若上传失败，重试次数
        threads: 1,                               //线程数量，考虑到服务器，这里就选了1
        duplicate: true,                          //分片是否自动去重
        formData: {                               //每次上传分片，一起携带的数据
            task_id: task_id,
        },
    });

    uploader.on('startUpload', function() {       //开始上传时，调用该方法
        $('.progress-bar').css('width', '0%');
        $('.progress-bar').text('0%');
    });

    uploader.on('uploadProgress', function(file, percentage) { //一个分片上传成功后，调用该方法
        $('.progress-bar').css('width', percentage * 100 - 1 + '%');
        $('.progress-bar').text(Math.floor(percentage * 100 - 1) + '%');
    });

    uploader.on('uploadSuccess', function(file) { //整个文件的所有分片都上传成功，调用该方法
        //上传的信息（文件唯一标识符，文件名）
        var data = {'task_id': task_id, 'filename': file.source['name'] };
        var aobj = document.getElementById("showbutton");
        var hrefobj = aobj.attributes["href"].value;
        $.get('/upload_file', data);          //ajax携带data向该url发请求
        $('.progress-bar').css('width', '100%').css('background-color','greenyellow');
        $('.progress-bar').text('上传完成');
        $('.turntoshow').attr("style","display:inline;");
        $("#showbutton").attr("value",data["filename"]).attr("href",hrefobj+data["filename"]);

    });

    uploader.on('uploadError', function(file) {   //上传过程中发生异常，调用该方法
        $('.progress-bar').css('width', '100%').css('background-color','red');
        $('.progress-bar').text('上传失败');
    });

    uploader.on('uploadComplete', function(file) {//上传结束，无论文件最终是否上传成功，该方法都会被调用
        $('.progress-bar').removeClass('active progress-bar-striped');
    });

});