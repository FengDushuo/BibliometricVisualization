var canGetCookie = 0;//是否支持存储Cookie 0 不支持 1 支持
var ajaxmockjax = 0;//是否启用虚拟Ajax的请求响 0 不启用  1 启用
//默认账号密码

var truelogin = "admin";
var truepwd = "123456";

var CodeVal = 0;

var showHelp = 1;

var nowtitle = document.title;

Code();
function Code() {
	if (canGetCookie == 1) {
		createCode("AdminCode");
		var AdminCode = getCookieValue("AdminCode");
		showCheck(AdminCode);
	} else {
		showCheck(createCode(""));
	}
}
function showCheck(a) {
	CodeVal = a;
	var c = document.getElementById("myCanvas");
	var ctx = c.getContext("2d");
	ctx.clearRect(0, 0, 1000, 1000);
	ctx.font = "80px 'Hiragino Sans GB'";
	ctx.fillStyle = "#E8DFE8";
	ctx.fillText(a, 0, 100);
}
$(document).keypress(function (e) {
	// 回车键事件  
	if (e.which == 13) {
		$('input[type="button"]').click();
	}
});

//粒子背景特效
$('body').particleground({
	dotColor: '#E8DFE8',
	lineColor: '#133b88'
});
$('input[name="pwd"]').focus(function () {
	$(this).attr('type', 'password');
});
$('input[type="text"]').focus(function () {
	$(this).prev().animate({ 'opacity': '1' }, 200);
});
$('input[type="text"],input[type="password"]').blur(function () {
	$(this).prev().animate({ 'opacity': '.5' }, 200);
});
$('input[name="login"],input[name="pwd"]').keyup(function () {
	var Len = $(this).val().length;
	if (!$(this).val() == '' && Len >= 5) {
		$(this).next().animate({
			'opacity': '1',
			'right': '30'
		}, 200);
	} else {
		$(this).next().animate({
			'opacity': '0',
			'right': '20'
		}, 200);
	}
});
var open = 0;
layui.use('layer', function () {
	if (nowtitle == "登录界面" && showHelp == 1 ) {
		var msgalert = '测试账号:' + truelogin + '<br/> 帐号密码:' + truepwd;
		var index = layer.alert(msgalert, { icon: 6, time: 4000, offset: 't', closeBtn: 0, title: '友情提示', btn: [], anim: 2, shade: 0 });
		layer.style(index, {
			color: '#777'
		});
		//非空验证
		$('input[id = "login"]').click(function () {
			var login = $('input[name="login"]').val();
			var pwd = $('input[name="pwd"]').val();
			var code = $('input[name="code"]').val();
			if (login == '') {
				ErroAlert('请输入您的账号');
			} else if (pwd == '') {
				ErroAlert('请输入密码');
			} else if (code == '' || code.length != 4) {
				ErroAlert('输入验证码');
			} else if (code.toUpperCase() != CodeVal.toUpperCase()) {
				ErroAlert('验证码错误');
			} else {
				//认证中..
				// fullscreen();
				//全屏
				$('.login').addClass('test'); //倾斜特效
				setTimeout(function () {
					$('.login').addClass('testtwo'); //平移特效
				}, 300);
				setTimeout(function () {
					$('.authent').show().animate({ right: -320 }, {
						easing: 'easeOutQuint',
						duration: 600,
						queue: false
					});
					$('.authent').animate({ opacity: 1 }, {
						duration: 200,
						queue: false
					}).addClass('visible');
				}, 500);

				//登陆
				var JsonData = { "username": login, "password": pwd };
				//此处做为ajax内部判断
				var url = "/";

				AjaxPost(url, JsonData,
					function () {
						//ajax加载中
					},
					function (data) {
						//ajax返回 
						//认证完成
						setTimeout(function () {
							$('.authent').show().animate({ right: 90 }, {
								easing: 'easeOutQuint',
								duration: 600,
								queue: false
							});
							$('.authent').animate({ opacity: 0 }, {
								duration: 200,
								queue: false
							}).addClass('visible');
							$('.login').removeClass('testtwo'); //平移特效
						}, 2000);
						setTimeout(function () {
							$('.authent').hide();
							$('.login').removeClass('test');
							if (data.info == "passworderror") {
								ErroAlert('密码错误');
							} else if (data.info == "user_not_exist") {
								ErroAlert('该用户不存在，请注册');
							} else if (data.info == "ok") {
								//登录成功
								$('.login div').fadeOut(100);
								$('.success').fadeIn(1000);
								$('.success').html("登陆成功<br /><br />欢迎回来");
								$('.success').fadeOut(3000);
								function to_user() {
									window.location.href = "/user";
								}
								setTimeout(to_user, 3000);

								//跳转操作

							} else {
								AjaxErro(data);
							}
						}, 2400);
					})
			}
		})
	}
	else if (nowtitle == "注册界面"){
		//非空验证
		$('input[id = "register"]').click(function () {
			var register = $('input[name="register"]').val();
			var email = $('input[name="email"]').val();
			var pwd = $('input[name="pwd"]').val();
			var rpwd = $('input[name="rpwd"]').val();
			var code = $('input[name="code"]').val();
			var accept = $("#accept-terms").is(":checked");
			if (register == '') {
				ErroAlert('请输入您的账号');
			} else if (email == '') {
				ErroAlert('请输入邮箱');
			} else if (pwd == '') {
				ErroAlert('请输入密码');
			} else if (rpwd == '') {
				ErroAlert('请确认密码');
			}else if (code == '' || code.length != 4) {
				ErroAlert('输入验证码');
			} else if (code.toUpperCase() != CodeVal.toUpperCase()) {
				ErroAlert('验证码错误');
			} else {
				//认证中..
				//fullscreen();
				//全屏
				$('.login').addClass('test'); //倾斜特效
				setTimeout(function () {
					$('.login').addClass('testtwo'); //平移特效
				}, 300);
				setTimeout(function () {
					$('.authent').show().animate({ right: -320 }, {
						easing: 'easeOutQuint',
						duration: 600,
						queue: false
					});
					$('.authent').animate({ opacity: 1 }, {
						duration: 200,
						queue: false
					}).addClass('visible');
				}, 500);
				//注册
				var JsonData = { "rusername": register, "rpassword": pwd, "rrepassword": rpwd, "remail": email, "accept":true };//有用户协议的时候可改成"accept":accept,在register.html中将注释的accept恢复
				//此处做为ajax内部判断
				var url = "/register";

				AjaxPost(url, JsonData,
					function () {
						//ajax加载中
					},
					function (data) {
						//ajax返回 
						//认证完成
						setTimeout(function () {
							$('.authent').show().animate({ right: 90 }, {
								easing: 'easeOutQuint',
								duration: 600,
								queue: false
							});
							$('.authent').animate({ opacity: 0 }, {
								duration: 200,
								queue: false
							}).addClass('visible');
							$('.login').removeClass('testtwo'); //平移特效
						}, 2000);
						setTimeout(function () {
							$('.authent').hide();
							$('.login').removeClass('test');
							if (data.info == "repassword_is_false!") {
								ErroAlert("确认密码错误，请重新输入");
							} else if (data.info == "do_not_full!") {
								ErroAlert("请补充好信息！");
							} else if (data.info == "user_is_exist!") {
								ErroAlert("该用户已存在，请登录！");
							} else if (data.info == "email_has_been_used!") {
								ErroAlert("邮箱已被注册，请更换邮箱！");
							} else if (data.info == "not_checked!") {
								ErroAlert("请同意用户协议！");
							} else if (data.info == "success!") {
								//登录成功
								$('.login div').fadeOut(100);
								$('.success').fadeIn(1000);
								$('.success').html("注册成功<br /><br />欢迎登录");
								$('.success').fadeOut(3000);
								function to_user() {
									showHelp = 0;
									window.location.href = "/";
								}
								setTimeout(to_user, 3000);

								//跳转操作

							} else {
								AjaxErro(data);
							}
						}, 2400);
					})
				}
			})
		}
	
	else if (nowtitle == "找回密码"){
		//非空验证
		$('input[id = "sendemail"]').click(function () {
			var user = $('input[name="login"]').val();
			var email = $('input[name="email"]').val();
			var domain = document.domain;
			var code = $('input[name="code"]').val();
			if (user == '') {
				ErroAlert('请输入您的账号');
			} else if (email == '') {
				ErroAlert('请输入邮箱');
			} else if (code == '' || code.length != 4) {
				ErroAlert('输入验证码');
			} else if (code.toUpperCase() != CodeVal.toUpperCase()) {
				ErroAlert('验证码错误');
			} else {
				//认证中..
				//fullscreen();
				//全屏
				$('.login').addClass('test'); //倾斜特效
				setTimeout(function () {
					$('.login').addClass('testtwo'); //平移特效
				}, 300);
				setTimeout(function () {
					$('.authent').show().animate({ right: -320 }, {
						easing: 'easeOutQuint',
						duration: 600,
						queue: false
					});
					$('.authent').animate({ opacity: 1 }, {
						duration: 200,
						queue: false
					}).addClass('visible');
				}, 500);
				//注册
				var JsonData = { "domain": domain, "email": email, };
				//此处做为ajax内部判断
				var url = "/forgetpwd";

				AjaxPost(url, JsonData,
					function () {
						//ajax加载中
					},
					function (data) {
						//ajax返回 
						//认证完成
						setTimeout(function () {
							$('.authent').show().animate({ right: 90 }, {
								easing: 'easeOutQuint',
								duration: 600,
								queue: false
							});
							$('.authent').animate({ opacity: 0 }, {
								duration: 200,
								queue: false
							}).addClass('visible');
							$('.login').removeClass('testtwo'); //平移特效
						}, 2000);
						setTimeout(function () {
							$('.authent').hide();
							$('.login').removeClass('test');
							if (data.info == "no_email_exist") {
								ErroAlert("输入的邮箱没有注册，请检查是否输错！");
							} else if (data.info == "no_email_here!") {
								ErroAlert("如需找回密码，请输入您的邮箱！");
							} else if (data.info == "success_send_email") {
								//登录成功
								$('.login div').fadeOut(100);
								$('.success').fadeIn(1000);
								$('.success').html("找回密码邮件发送成功  请查收<br /><br />如果没收到提示请检查邮箱垃圾箱！");
								$('.success').fadeOut(4000);
								function to_user() {
									showHelp = 0;
									window.location.href = "/";
								}
								setTimeout(to_user, 3000);

								//跳转操作
								
							} else {
								AjaxErro(data);
							}
						}, 2400);
					})
				}
			})
		}

	else if (nowtitle == "重置密码"){
		//非空验证
		$('input[id = "reset"]').click(function () {
			var user = $('input[name="login"]').val();
			var email = $('input[name="email"]').val();
			var password = $('input[name="pwd"]').val();
			var rpassword = $('input[name="rpwd"]').val();
			var code = $('input[name="code"]').val();
			if (user == '') {
				ErroAlert('请输入您的账号');
			} else if (email == '') {
				ErroAlert('请输入邮箱');
			} else if (password == '') {
				ErroAlert('请输入密码');
			} else if (rpassword == '') {
				ErroAlert('请输入确认密码');
			} else if (code == '' || code.length != 4) {
				ErroAlert('输入验证码');
			} else if (code.toUpperCase() != CodeVal.toUpperCase()) {
				ErroAlert('验证码错误');
			} else {
				//认证中..
				//fullscreen();
				//全屏
				$('.login').addClass('test'); //倾斜特效
				setTimeout(function () {
					$('.login').addClass('testtwo'); //平移特效
				}, 300);
				setTimeout(function () {
					$('.authent').show().animate({ right: -320 }, {
						easing: 'easeOutQuint',
						duration: 600,
						queue: false
					});
					$('.authent').animate({ opacity: 1 }, {
						duration: 200,
						queue: false
					}).addClass('visible');
				}, 500);
				//重置密码
				var JsonData = { "username":user, "password":password, "repassword":rpassword,"email":email };
				//此处做为ajax内部判断
				var url = "/modify";

				AjaxPost(url, JsonData,
					function () {
						//ajax加载中
					},
					function (data) {
						//ajax返回 
						//认证完成
						setTimeout(function () {
							$('.authent').show().animate({ right: 90 }, {
								easing: 'easeOutQuint',
								duration: 600,
								queue: false
							});
							$('.authent').animate({ opacity: 0 }, {
								duration: 200,
								queue: false
							}).addClass('visible');
							$('.login').removeClass('testtwo'); //平移特效
						}, 2000);
						setTimeout(function () {
							$('.authent').hide();
							$('.login').removeClass('test');
							if (data.info == "repassword_is_false!") {
								ErroAlert("确认密码错误，请重新输入！");
							} else if (data.info == "do_not_full!") {
								ErroAlert("请补充好信息！");
							} else if (data.info == "username_email_unfit!") {
								ErroAlert("用户名与邮箱不匹配！请重新输入！");
							} else if (data.info == "username_not_exist!") {
								ErroAlert("用户不存在，请返回注册！");
							} else if (data.info == "success!") {
								//登录成功
								$('.login div').fadeOut(100);
								$('.success').fadeIn(1000);
								$('.success').html("密码重置成功<br /><br />请您返回登录");
								$('.success').fadeOut(4000);
								function to_user() {
									showHelp = 0;
									window.location.href = "/";
								}
								setTimeout(to_user, 3000);

								//跳转操作
								
							} else {
								AjaxErro(data);
							}
						}, 2400);
					})
				}
			})
		}

})
var fullscreen = function () {
	elem = document.body;
	if (elem.webkitRequestFullScreen) {
		elem.webkitRequestFullScreen();
	} else if (elem.mozRequestFullScreen) {
		elem.mozRequestFullScreen();
	} else if (elem.requestFullScreen) {
		elem.requestFullscreen();
	} else {
		//浏览器不支持全屏API或已被禁用  
	}
}
if (ajaxmockjax == 1) {
	$.mockjax({
		url: 'Ajax/Login',
		status: 200,
		responseTime: 50,
		responseText: { "Status": "ok", "Text": "登陆成功<br /><br />欢迎回来" }
	});
	$.mockjax({
		url: 'Ajax/LoginFalse',
		status: 200,
		responseTime: 50,
		responseText: { "Status": "Erro", "Erro": "账号名或密码或验证码有误" }
	});
}