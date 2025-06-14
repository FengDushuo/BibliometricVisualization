
  var param_json = JSON.parse(document.getElementById("global_data_json").value)
  var files_path = param_json.data_path
  var timeline = param_json.timelines[0].toString()
  var file = files_path + timeline+'-keyword.json';
  var count_file = files_path + 'count_timeline.json';

  var timeline1 = timeline;
  
  var flag=0;

  const order = "desc";
  const width1 = 460;
  const height1 = 460;

  var researches = param_json.researches
  var colors_in_circle = param_json.colors_in_circle
  var colors1 = {}
  researches.forEach((key, index) => {
    colors1[key] = colors_in_circle[index];
  });
  var type = researches[0];
  var type1 = researches[0];


function suiJi(m,n){
  return m+parseInt(Math.random()*(n-m+1))
}

function yanSe(){
  var result = "#"
  for(var i = 0; i<6; i++){
      result +=suiJi(0,15).toString(16)
  }
  return result;
  //生成一个随机颜色编码#000000-#ffffff
}



function mousePosition(ev){ 
    ev = ev || window.event; 
    if(ev.pageX || ev.pageY){ 
        return {x:ev.pageX, y:ev.pageY}; 
    } 
    return { 
        x:ev.clientX + document.body.scrollLeft - document.body.clientLeft, 
        y:ev.clientY + document.body.scrollTop - document.body.clientTop 
    }; 
}   

function generateChart(data) {
  var myChart = echarts.init(document.getElementById('bubble-chart'));
  console.log(data)
  // ECharts 配置项
  var option = {
    tooltip: {
        show: true
    },
    series: [{
        type: 'wordCloud',
        shape: 'circle',
        keepAspect: false,
      // maskImage: maskImage,
        left: 'center',
        top: 'center',
        width: '80%',
        height: '80%',
        right: null,
        bottom: null,
        sizeRange: [5, 20],
        rotationRange: [0, 0],
        rotationStep: 0,
        gridSize: 1,
        drawOutOfBound: true,
        shrinkToFit: true,
        layoutAnimation: true,
        textStyle: {
            fontFamily: 'sans-serif',
            fontWeight: 'bold',
            color: function () {
                return 'rgb(' + [
                    Math.round(Math.random() * 160),
                    Math.round(Math.random() * 160),
                    Math.round(Math.random() * 160)
                ].join(',') + ')';
            }
        },
        emphasis: {
            // focus: 'self',
            textStyle: {
                textShadowBlur: 3,
                textShadowColor: '#333'
            }
        },
        data: data
    }]
};

// 使用指定的配置项和数据显示图表
myChart.setOption(option);
};

function generateChart1(data) {
  var myChart = echarts.init(document.getElementById('bubble-chart1'));
  // ECharts 配置项
  var option = {
      tooltip: {
          show: true
      },

      series: [{
          type: 'wordCloud',
          shape: 'circle',
          keepAspect: false,
        // maskImage: maskImage,
          left: 'center',
          top: 'center',
          width: '100%',
          height: '90%',
          right: null,
          bottom: null,
          sizeRange: [5, 20],
          rotationRange: [0, 0],
          rotationStep: 0,
          gridSize: 1,
          shrinkToFit: true,
          drawOutOfBound: false,
          layoutAnimation: true,
          textStyle: {
              fontFamily: 'sans-serif',
              fontWeight: 'bold',
              color: function () {
                  return 'rgb(' + [
                      Math.round(Math.random() * 160),
                      Math.round(Math.random() * 160),
                      Math.round(Math.random() * 160)
                  ].join(',') + ')';
              }
          },
          emphasis: {
              // focus: 'self',
              textStyle: {
                  textShadowBlur: 3,
                  textShadowColor: '#333'
              }
          },
          data: data
      }]
  };

  // 使用指定的配置项和数据显示图表
  myChart.setOption(option);
};

(async () => {
  d3.json(file).then(function(data) {
    // 提取词云数据
    console.log(data);
    var wordArray= data.map(d => ({
        name: d.keyword,
        value: d.count // 调整字体大小因子
    }));
    generateChart(wordArray);
  });
  
})();


  var width = 540;
  var height = 540;
  var minOfWH = Math.min(width, height) / 2 +100;
  var initialAnimDelay = 300;
  var arcAnimDelay = 150;
  var arcAnimDur = 1000;
  var secDur = 1000;
  var secIndividualdelay = 150;

  var radius = void 0;

  // calculate minimum of width and height to set chart radius
  if (minOfWH > 380) {
    radius = 380;
  } else {
    radius = minOfWH;
  }

  var draw = function draw() {

    var  emblemText= ['K E Y W O R D S', 'K E Y W O R D S', 'K E Y W O R D S'];    
    circleText('.emblem');
    function   circleText (el, str) {
      let element = document.querySelector(el);
      emblemText.forEach((v, i) => {
        let span = document.createElement('span');
        span.innerHTML = `<pre style="color:black;font-weight:bold;">${emblemText[i]}</pre>`
        let deg = (i) * 120 + 60;
        span.style.transform = `rotateZ(${deg}deg)`
        element.appendChild(span);
      })
    };
    const splitLongString = (str, count) => {
      const partLength = Math.round(str.length / count);
      const words = str.split(' ');
      const parts = [];
      str.split(' ').forEach(part => {
      if (!parts.length) {
        parts.push(part);
      }
      else {
        const last = parts[parts.length - 1];
        if (parts[parts.length - 1].length >= partLength)
        parts.push(part);
      else  
        parts[parts.length - 1] += ' ' + part;
      }
    });
    return parts;
    };

    var myChart = echarts.init(document.getElementById('onecircle'));
      const value = 0.25 // 外层圆的数值
      myChart.setOption({
            series: [
              { // 内部的饼图
                type: 'pie',
                radius: ['65%', '67%'],
                avoidLabelOverlap: false,
                emphasis: {
                  // disabled: true
                  label: {
                    
                  }
                },
                labelLine: {
                  show: false
                },
                label: {
                  show: false,
                  position: 'center',
                  rich: {
                    a: {
                      fontSize: 72,
                      color: '#fff',
                      textAlign: 'center',
                      padding: [0, 0, 12, 0]
                    },
                    b: {
                      padding: [8, 6, 6, 6],
                      fontSize: 20,
                      color: '#666',
                      verticalAlign: 'top'
                    },
                    c: {
                      width: '100%',
                      height: 2,
                      backgroundColor: '#dadde4',
                      align: 'left'
                    },
                    d: {
                      fontSize: 36,
                      color: '#999',
                      padding: [20, 0, 12, 0]
                    }
                  }
                },
                data: [
                  {
                    value: 45, 
                    name: 'Search Engine',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  }
                ]
              }
            ]     
      });

      var colors_dataq = param_json.colors_dataq
      var researches = param_json.researches

      var dataq = colors_dataq.map((name, index) =>{
        return {
          value:1,
          text:researches[index],
          color:colors1[researches[index]]
        };
      });
  
    // const dataq = [
      
    //   {value: 1, text: "Computer Science Information Systems", color: '#666666'},
    //   {value: 1, text: "Computer Science Artificial Intelligence", color: '#006634'},
    //   {value: 1, text: "Telecommunications", color: '#66999A'},
    //   {value: 1, text: "Sport Sciences", color: '#FEE100'},
    //   {value: 1, text: "Chemistry Analytical", color: '#FF7F00'},
    //   {value: 1, text: "Multidisciplinary Sciences", color: '#6599FF'},
    //   {value: 1, text: "PhysicsApplied", color: '#999999'},
    //   {value: 1, text: "Engineering Electrical Electronic", color: '#99CCCD'}];
    const svg = d3.select('.chart-wrapper')
    .select('.pieChart')
    .style('width', width)
    .style('height', height);
    const margin = 5;
    const arcWidth = 35;
    const radius = Math.min(width/2 - margin, height/2 - margin) - arcWidth / 2;
    const center = {x: width / 2, y: height / 2};

    let anglePos = 0;
    const angleOffset = 0.005;

    const sum = dataq.reduce((s, {value}) => s + value, 0);
    dataq.forEach(({value, text, color}, index) => {
      const angle = Math.PI * 2 * value / sum;
      const startAngle = anglePos + angleOffset;
      anglePos += angle;
      const endAngle = anglePos - angleOffset;
      const start = {
        x: center.x + radius * Math.sin(startAngle),
        y: center.y + radius * -Math.cos(startAngle),
      };
      const end = {
        x: center.x + radius * Math.sin(endAngle),
        y: center.y + radius * -Math.cos(endAngle),
      };
      const flags = value / sum >= 0.5 ? '1 1 1' : '0 0 1';
      const pathId = `my-pie-chart-path-${index}`;
      const path = svg.append('path')
        .attr('id', pathId)
        .attr('d', `M ${start.x},${start.y} A ${radius},${radius} ${flags} ${end.x},${end.y}`)
        .style('stroke', color)
        .style('fill', 'none')
        .style('stroke-width', arcWidth);
        svg.selectAll("path")
        .on('click', function (e) {piesingle(e)})
        .on('dblclick', function (e) {piedouble(e)});
        
      const len = path.node().getTotalLength();
      
      const textElement = svg.append('text')
        .text(text)
        .attr('dy', 0)
        .attr('text-anchor', 'middle')
        .attr("font-size",50)
        .attr("font-family", "Microsoft Yahei")
        .style('fill', '#000')
        .style('opacity', 1)
        ;
      const width = textElement.node().getBBox().width;  
      let texts = [text];
      if (width > len)
        texts = splitLongString(text, Math.ceil(width / len));
            
      textElement.text(null);
      
      // const midAngle = anglePos - angle / 2;
      
      texts.forEach((t, i) => {
        const textPathId = `my-pie-chart-path-${index}-${i}`;
        const textRadius = radius - i * 12;
        const textStart = {
          x: center.x + textRadius * Math.sin(startAngle),
          y: center.y + textRadius * -Math.cos(startAngle),
        };
        const textEnd = {
          x: center.x + textRadius * Math.sin(endAngle),
          y: center.y + textRadius * -Math.cos(endAngle),
        };

        const path = svg.append('path')
          .attr('id', textPathId)
          .attr('d', `M ${textStart.x},${textStart.y} A ${textRadius},${textRadius} ${flags} ${textEnd.x},${textEnd.y}`)
          .style('stroke', 'none')
          .style('fill', 'none')
          ;
        
        textElement.append('textPath')
          .text(t)
          .attr('startOffset', (endAngle - startAngle) * textRadius / 2)
          .attr('href', `#${textPathId}`)
      });

    });
  };

  draw();

  var time = 200; //300以上，双击才生效
  var timeOut = null;

  function single (e) {
      console.log(e)
      clearTimeout(timeOut); // 清除第一个单击事件
      timeOut = setTimeout(function () {
          console.log('单击');
          const svg = d3.select('#bubble-chart');
          svg.selectAll("g").remove()
          .transition()
          .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;}).
          duration(secDur);
          file = files_path+e.data.name+'-keyword.json'; 
          timeline = e.data.name;
          var names1 = []; //类别数组（用于存放饼图的类别）
          var brower1 = [];
          $.ajax({
            url: files_path+"count_timeline.json",
            data: {},
            type: 'GET',
            success: function(data) {
                //请求成功时执行该函数内容，result即为服务器返回的json对象
                $.each(data, function(index, item) {
                    names1.push(item.value); //挨个取出类别并填入类别数组
                    brower1.push({
                        name: item.timeline,
                        value: item.infected
                    });
                });
                hrFun(brower1);
            },
          });
          (async () => {
            // var data = await d3.json(file).then(data => data);
            // generateChart(data);
            d3.json(file).then(function(data) {
              // 提取词云数据
              var wordArray= data.map(d => ({
                  name: d.keyword,
                  value: d.count // 调整字体大小因子
              }));
              generateChart(wordArray);
            });
            
          })();
          // 单击事件的代码执行区域
          // ...
      }, time)
  }
  function double (e) {
      flag+=1;
      // var traget=document.getElementById('close');
      // traget.style.display="block";
      clearTimeout(timeOut); // 清除第二个单击事件
      const svg = d3.select('#bubble-chart1');
      svg.selectAll("g").remove()
      .transition()
      .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
      .duration(secDur);
      var file1 = files_path+e.data.name+'-keyword.json'; 
      timeline1 = e.data.name
      var names1 = []; //类别数组（用于存放饼图的类别）
      var brower1 = [];
      $.ajax({
        url: files_path+"count_timeline.json",
        data: {},
        type: 'GET',
        success: function(data) {
            //请求成功时执行该函数内容，result即为服务器返回的json对象
            $.each(data, function(index, item) {
                names1.push(item.value); //挨个取出类别并填入类别数组
                brower1.push({
                    name: item.timeline,
                    value: item.infected
                });
            });
            hrFun2(brower1);
        },
      });
      (async () => {
        if (flag%2==1){
          var traget=document.getElementById('main1');
          traget.style.display="block";
          d3.select('#bubble-chart1').selectAll("g").style('opacity',1);
          d3.select('.pieChart1').selectAll("g").style('opacity',1);
          // var data = await d3.json(file1).then(data => data);
          // generateChart1(data);
          d3.json(file1).then(function(data) {
            // 提取词云数据
            var wordArray= data.map(d => ({
                name: d.keyword,
                value: d.count // 调整字体大小因子
            }));
            generateChart1(wordArray);
          });
          
          draw1();
        }else{
          var traget=document.getElementById('main1');
          traget.style.display="none";
          d3.select('#bubble-chart1').selectAll("g").remove();
          d3.select('.pieChart1').selectAll("g").remove()
        }
      })();
      // 双击的代码执行区域
      // ...
  }

  function single1 (e) {
    console.log(e)
    clearTimeout(timeOut); // 清除第一个单击事件
    timeOut = setTimeout(function () {
        console.log('单击');
        const svg = d3.select('#bubble-chart1');
        svg.selectAll("g").remove()
        .transition()
        .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
        .duration(secDur);
        file = files_path+e.data.name+'-keyword.json'; 
        timeline1 = e.data.name;
        var names1 = []; //类别数组（用于存放饼图的类别）
        var brower1 = [];
        $.ajax({
          url: files_path+"count_timeline.json",
          data: {},
          type: 'GET',
          success: function(data) {
              //请求成功时执行该函数内容，result即为服务器返回的json对象
              $.each(data, function(index, item) {
                  names1.push(item.value); //挨个取出类别并填入类别数组
                  brower1.push({
                      name: item.timeline,
                      value: item.infected
                  });
              });
              hrFun2(brower1);
          },
        });
        (async () => {
          // var data = await d3.json(file).then(data => data);
          // generateChart1(data);
          d3.json(file).then(function(data) {
            // 提取词云数据
            var wordArray= data.map(d => ({
                name: d.keyword,
                value: d.count // 调整字体大小因子
            }));
            generateChart1(wordArray);
          });
            
        })();
        // 单击事件的代码执行区域
        // ...
    }, time)
}
function double1 (e) {
    clearTimeout(timeOut); // 清除第二个单击事件
    timeline1 = e.data.name
    var file1 = files_path+e.data.name+'-keyword.json'; 
    var names1 = []; //类别数组（用于存放饼图的类别）
    var brower1 = [];
    const svg = d3.select('#bubble-chart');
    svg.selectAll("g").remove()
    .transition()
    .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
    .duration(secDur);
    $.ajax({
      url: files_path+"count_timeline.json",
      data: {},
      type: 'GET',
      success: function(data) {
          //请求成功时执行该函数内容，result即为服务器返回的json对象
          $.each(data, function(index, item) {
              names1.push(item.value); //挨个取出类别并填入类别数组
              brower1.push({
                  name: item.timeline,
                  value: item.infected
              });
          });
          hrFun(brower1);
      },
    });
    (async () => {
      // var data = await d3.json(file1).then(data => data);
      // generateChart(data);
      d3.json(file1).then(function(data) {
        // 提取词云数据
        var wordArray= data.map(d => ({
            name: d.keyword,
            value: d.count // 调整字体大小因子
        }));
        generateChart(wordArray);
      });    
      draw();
    })();
    // 双击的代码执行区域
    // ...
}


function piesingle (e) {
  console.log(e)
  clearTimeout(timeOut); // 清除第一个单击事件
  timeOut = setTimeout(function () {
      console.log('单击');
      const svg = d3.select('#bubble-chart');
      svg.selectAll("g").remove()
      .transition()
      .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
      .duration(secDur);
      file = files_path+timeline+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'-keyword.json'; 
      (async () => {
        // var data = await d3.json(file).then(data => data);
        // generateChart(data);
        d3.json(file).then(function(data) {
          // 提取词云数据
          var wordArray= data.map(d => ({
              name: d.keyword,
              value: d.count // 调整字体大小因子
          }));
          generateChart(wordArray);
        });
        
      })();
      // 单击事件的代码执行区域
      // ...
  }, time)
}
function piedouble (e) {
  flag+=1;
  clearTimeout(timeOut); // 清除第二个单击事件
  const svg = d3.select('#bubble-chart1');
  svg.selectAll("g").remove()
  .transition()
  .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
  .duration(secDur);
  var file1 = files_path+timeline+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'-keyword.json'; 
  var names1 = []; //类别数组（用于存放饼图的类别）
  var brower1 = [];
  $.ajax({
    url: files_path+"count_timeline.json",
    data: {},
    type: 'GET',
    success: function(data) {
        //请求成功时执行该函数内容，result即为服务器返回的json对象
        $.each(data, function(index, item) {
            names1.push(item.value); //挨个取出类别并填入类别数组
            brower1.push({
                name: item.timeline,
                value: item.infected
            });
        });
        hrFun2(brower1);
    },
  });
  (async () => {
    if (flag%2==1){
      var traget=document.getElementById('main1');
      traget.style.display="block";
      d3.select('#bubble-chart1').selectAll("g").style('opacity',1);
      d3.select('.pieChart1').selectAll("g").style('opacity',1);
      // var data = await d3.json(file1).then(data => data);
      // generateChart1(data);
      d3.json(file1).then(function(data) {
        // 提取词云数据
        var wordArray= data.map(d => ({
            name: d.keyword,
            value: d.count // 调整字体大小因子
        }));
        generateChart1(wordArray);
      });
      
      draw1();
    }else{
      var traget=document.getElementById('main1');
      traget.style.display="none";
      d3.select('#bubble-chart1').selectAll("g").remove();
      d3.select('.pieChart1').selectAll("g").remove()
    }
  })();
  // 双击的代码执行区域
  // ...
}

function piesingle1 (e) {
console.log(e)
clearTimeout(timeOut); // 清除第一个单击事件
timeOut = setTimeout(function () {
    console.log('单击');
    const svg = d3.select('#bubble-chart1');
    svg.selectAll("g").remove()
    .transition()
    .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
    .duration(secDur);
    file = files_path+timeline1+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'-keyword.json'; 
    (async () => {
      // var data = await d3.json(file).then(data => data);
      // generateChart1(data);
      d3.json(file).then(function(data) {
        // 提取词云数据
        var wordArray= data.map(d => ({
            name: d.keyword,
            value: d.count // 调整字体大小因子
        }));
        generateChart1(wordArray);
      });
      
    })();
    // 单击事件的代码执行区域
    // ...
}, time)
}
function piedouble1 (e) {
clearTimeout(timeOut); // 清除第二个单击事件
var file1 = files_path+timeline1+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'-keyword.json'; 
var names1 = []; //类别数组（用于存放饼图的类别）
var brower1 = [];
const svg = d3.select('#bubble-chart');
svg.selectAll("g").remove()
.transition()
.delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
.duration(secDur);
$.ajax({
  url: files_path+"count_timeline.json",
  data: {},
  type: 'GET',
  success: function(data) {
      //请求成功时执行该函数内容，result即为服务器返回的json对象
      $.each(data, function(index, item) {
          names1.push(item.value); //挨个取出类别并填入类别数组
          brower1.push({
              name: item.timeline,
              value: item.infected
          });
      });
      hrFun1(brower1);
  },
});
(async () => {
  // var data = await d3.json(file1).then(data => data);
  // generateChart(data);
  d3.json(file1).then(function(data) {
    // 提取词云数据
    var wordArray= data.map(d => ({
        name: d.keyword,
        value: d.count // 调整字体大小因子
    }));
    generateChart(wordArray);
  });
  
  draw();
})();
// 双击的代码执行区域
// ...
}

  var draw1 = function draw1() {
    
    var  emblemText=  ['K E Y W O R D S', 'K E Y W O R D S', 'K E Y W O R D S'];   
    circleText('.emblem1');
    function   circleText (el, str) {
      let element = document.querySelector(el);
      emblemText.forEach((v, i) => {
        let span = document.createElement('span');
        span.innerHTML = `<pre style="color:black;font-weight:bold;">${emblemText[i]}</pre>`
        let deg = (i) * 120 + 60;
        span.style.transform = `rotateZ(${deg}deg)`
        element.appendChild(span);
      })
    };
    var emblem1 = document.querySelector('.emblem1');
    emblem1.style.display='inline'
    const splitLongString = (str, count) => {
      const partLength = Math.round(str.length / count);
      const words = str.split(' ');
      const parts = [];
      str.split(' ').forEach(part => {
      if (!parts.length) {
        parts.push(part);
      }
      else {
        const last = parts[parts.length - 1];
        if (parts[parts.length - 1].length >= partLength)
        parts.push(part);
      else  
        parts[parts.length - 1] += ' ' + part;
      }
    });
    return parts;
    };

    var myChart = echarts.init(document.getElementById('onecircle1'));
      const value = 0.25 // 外层圆的数值
      myChart.setOption({
            series: [
              { // 内部的饼图
                type: 'pie',
                radius: ['65%', '67%'],
                avoidLabelOverlap: false,
                emphasis: {
                  // disabled: true
                  label: {
                    
                  }
                },
                labelLine: {
                  show: false
                },
                label: {
                  show: false,
                  position: 'center',
                  rich: {
                    a: {
                      fontSize: 72,
                      color: '#fff',
                      textAlign: 'center',
                      padding: [0, 0, 12, 0]
                    },
                    b: {
                      padding: [8, 6, 6, 6],
                      fontSize: 20,
                      color: '#666',
                      verticalAlign: 'top'
                    },
                    c: {
                      width: '100%',
                      height: 2,
                      backgroundColor: '#dadde4',
                      align: 'left'
                    },
                    d: {
                      fontSize: 36,
                      color: '#999',
                      padding: [20, 0, 12, 0]
                    }
                  }
                },
                data: [
                  {
                    value: 45, 
                    name: 'Search Engine',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Direct',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 2,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#fff'
                    }
                  },
                  { 
                    value: 45,
                    name: 'Email',
                    itemStyle: {
                      borderCap: 'round',
                      borderRadius: '100%',
                      borderWidth: 6,	//边框的宽度
                      borderColor: '#fff',	//边框的颜色
                      color:'#000'
                    }
                  }
                ]
              }
            ]     
      });

    var colors_dataq = param_json.colors_dataq
    var researches = param_json.researches

    var dataq = colors_dataq.map((name, index) =>{
      return {
        value:1,
        text:researches[index],
        color:colors1[researches[index]]
      };
    });
  
    const svg = d3.select('.chart-wrapper1')
    .select('.pieChart1')
    .style('width', width)
    .style('height', height);
    const margin = 5;
    const arcWidth = 35;
    const radius = Math.min(width/2 - margin, height/2 - margin) - arcWidth / 2;
    const center = {x: width / 2, y: height / 2};

    let anglePos = 0;
    const angleOffset = 0.005;

    const sum = dataq.reduce((s, {value}) => s + value, 0);
    dataq.forEach(({value, text, color}, index) => {
      const angle = Math.PI * 2 * value / sum;
      const startAngle = anglePos + angleOffset;
      anglePos += angle;
      const endAngle = anglePos - angleOffset;
      const start = {
        x: center.x + radius * Math.sin(startAngle),
        y: center.y + radius * -Math.cos(startAngle),
      };
      const end = {
        x: center.x + radius * Math.sin(endAngle),
        y: center.y + radius * -Math.cos(endAngle),
      };
      const flags = value / sum >= 0.5 ? '1 1 1' : '0 0 1';
      const pathId = `my-pie-chart-path-${index}`;
      const path = svg.append('path')
        .attr('id', pathId)
        .attr('d', `M ${start.x},${start.y} A ${radius},${radius} ${flags} ${end.x},${end.y}`)
        .style('stroke', color)
        .style('fill', 'none')
        .style('stroke-width', arcWidth);
        svg.selectAll("path")
        .on('click', function (e) {piesingle1(e)})
        .on('dblclick', function (e) {piedouble1(e)});
        
      const len = path.node().getTotalLength();
      
      const textElement = svg.append('text')
        .text(text)
        .attr('dy', 0)
        .attr('text-anchor', 'middle')
        .attr("font-size",50)
        .attr("font-family", "Microsoft Yahei")
        .style('fill', '#000')
        .style('opacity', 1)
        ;
      const width = textElement.node().getBBox().width;  
      let texts = [text];
      if (width > len)
        texts = splitLongString(text, Math.ceil(width / len));
            
      textElement.text(null);
      
      // const midAngle = anglePos - angle / 2;
      
      texts.forEach((t, i) => {
        const textPathId = `my-pie-chart-path-${index}-${i}`;
        const textRadius = radius - i * 12;
        const textStart = {
          x: center.x + textRadius * Math.sin(startAngle),
          y: center.y + textRadius * -Math.cos(startAngle),
        };
        const textEnd = {
          x: center.x + textRadius * Math.sin(endAngle),
          y: center.y + textRadius * -Math.cos(endAngle),
        };

        const path = svg.append('path')
          .attr('id', textPathId)
          .attr('d', `M ${textStart.x},${textStart.y} A ${textRadius},${textRadius} ${flags} ${textEnd.x},${textEnd.y}`)
          .style('stroke', 'none')
          .style('fill', 'none');
        
        textElement.append('textPath')
          .text(t)
          .attr('startOffset', (endAngle - startAngle) * textRadius / 2)
          .attr('href', `#${textPathId}`)
      });

    });
    
  };

  (async () => {
    var data2 = await d3.json(count_file).then(data => data);
  })();

  var names = []; //类别数组（用于存放饼图的类别）

  var brower = [];
  $.ajax({
      url: files_path+"count_timeline.json",
      data: {},
      type: 'GET',
      success: function(data) {
          //请求成功时执行该函数内容，result即为服务器返回的json对象
          $.each(data, function(index, item) {
              names.push(item.value); //挨个取出类别并填入类别数组
              brower.push({
                  name: item.timeline,
                  value: item.infected
              });
          });
          hrFun(brower);
      },
  });
  // 基于准备好的dom，初始化echarts实例
  
  function hrFun(param) {
      var myChart = echarts.init(document.getElementById('main'));
      myChart.on('click', function (e) {
        single(e);
      });
      myChart.on('dblclick', function (e) {
        double(e);
      });
      var showData = [];
      var sum = 0, max =0;
      brower.forEach(item => {
          sum += item.value
          if(item.value >= max) max = item.value
      })
      // 放大规则
      var number = Math.round(max * 0.5)
      showData = brower.map(item => {
          return {
              value: item.value+number,
              name: item.name,
          }
      })
      var textdata = timeline;
      myChart.setOption({
          title: {
              show: true,
              text: textdata,
              x:'0', //水平安放位置，默认为'left'，可选为：'center' | 'left' | 'right' | {number}（x坐标，单位px）
              y: '0', //垂直安放位置，默认为top，可选为：'top' | 'bottom' | 'center' | {number}（y坐标，单位px）
              textStyle: {
                // 主标题文本样式
                fontFamily: "Microsoft Yahei",
                fontSize: 25,
                fontStyle: "normal",
                fontWeight: "bold",
                color: "#000",
              },
          },
          toolbox: {
              show: false,
              feature: {
                  mark: {
                      show: true
                  },
                  dataView: {
                      show: true,
                      readOnly: false
                  },
                  restore: {
                      show: true
                  },
                  saveAsImage: {
                      show: true
                  }
              }
          },
          tooltip: {
            show: true,    // 是否显示提示框组件
            trigger: 'item',    // 触发类型（'item'，数据项图形触发，主要在散点图，饼图等无类目轴的图表中使用；'axis'，坐标轴触发，主要在柱状图，折线图等会使用类目轴的图表中使用；'none'，不触发。）
            formatter: function (param){
                console.log(param.flag)
                return param.name +': '+ parseInt(param.value-number);
            },
            extraCssText: 'z-index: 9',
            showContent: true,     // 是否显示提示框浮层，默认显示
            alwaysShowContent: false,     // 是否永远显示提示框内容，默认情况下在移出可触发提示框区域后一定时间后隐藏
            triggerOn: 'mousemove|click',    // 提示框触发的条件（'mousemove'，鼠标移动时触发；'click'，鼠标点击时触发；'mousemove|click'，同时鼠标移动和点击时触发；'none'，不在 'mousemove' 或 'click' 时触发）
            confine: true,    // 是否将 tooltip 框限制在图表的区域内
            backgroundColor: 'rgba(229, 222, 222, 0.7)',    // 提示框浮层的背景颜色
            padding: 5,    // 提示框浮层内边距，单位px
            textStyle: {
                color: '#000',     // 文字的颜色
                fontStyle: 'normal',    // 文字字体的风格（'normal'，无样式；'italic'，斜体；'oblique'，倾斜字体） 
                fontWeight: 'bold',    // 文字字体的粗细（'normal'，无样式；'bold'，加粗；'bolder'，加粗的基础上再加粗；'lighter'，变细；数字定义粗细也可以，取值范围100至700）
                fontSize: '30',    // 文字字体大小
                lineHeight: '50',    // 行高 
            }
          },
          series: [{
              name: '数量',
              type: 'pie',
              radius: [360, 420],
              center: ['50%', '50%'],
              roseType: 'area',
              itemStyle: {
                  borderRadius: 2,
                  borderWidth:6,
                  borderColor:'#fff',
                  color:'#A9A9A9'
              },
              label: {
                  padding: [0,0,0,0],
                  normal: {
                    show: true,
                    position:'insideTop',
                    formatter: (params) => {
                      const { name, value } = params;
                      const maxLength = 12; // 每行最多字符数
                      const maxLines = 3;  // 限制最大行数                   
                      // 把名称分割成每行最多 12 个字符
                      let wrappedName = name.match(/.{1,12}/g)?.join('\n') || name;
                      // 如果行数超过最大行数，则截断并加上省略号
                      const lines = wrappedName.split('\n');
                      if (lines.length > maxLines) {
                          wrappedName = lines.slice(0, maxLines).join('\n') + '\n...';
                      }
                      return `${wrappedName || name}`;  // 显示名称部分
                  },
                      textStyle: { //标签的字体样式
                        color: '#000', //字体颜色
                        fontStyle: 'oblique',//文字字体的风格 'normal'标准 'italic'斜体 'oblique' 倾斜
                        fontWeight: 'bold',//'normal'标准'bold'粗的'bolder'更粗的'lighter'更细的或100 | 200 | 300 | 400...
                        fontFamily: 'Microsoft Yahei', //文字的字体系列
                        fontSize: 20, //字体大小
                      },
                      rotate: 'radial'
                  }
              },
              labelLine: {
                  normal: {
                      show: false
                  }
              },
              
              data: showData,
          },
          {
            name: '数量',
            type: 'pie',
            radius: [360, 420],
            center: ['50%', '50%'],
            roseType: 'area',
            label: {
                normal: {
                    show: true,
                    position: 'inside', // 将值显示在饼图的底部
                    formatter: function (params) {
                        const { value } = params;
                        return `${parseInt(value-number)}`; // 只显示值部分
                    },
                    textStyle: {
                        color: 'red',  // 控制value文本的颜色
                        fontSize: 18,   // 控制value文本的大小
                        fontWeight: 'bold',
                    }
                }
            },
            labelLine: {
                normal: {
                    show: false
                }
            },
            itemStyle: {
                borderRadius: 2,
                borderWidth: 6,
                borderColor: '#fff',
                color: '#A9A9A9'
            },
            data: showData,
        }
        ]
      });
  }

  function hrFun2(param) {
    var myChart = echarts.init(document.getElementById('main1'));
    myChart.on('click', function (e) {
      single1(e);
    });
    myChart.on('dblclick', function (e) {
      double1(e);
    });
    var showData = [];
    var sum = 0, max =0;
    brower.forEach(item => {
        sum += item.value
        if(item.value >= max) max = item.value
    })
    // 放大规则
    var number = Math.round(max * 0.5)
    showData = brower.map(item => {
        return {
            value: item.value+number,
            name: item.name,
        }
    })
    var textdata = timeline1;
    myChart.setOption({
        title: {
            show: true,
            text: textdata,
            x:'0', //水平安放位置，默认为'left'，可选为：'center' | 'left' | 'right' | {number}（x坐标，单位px）
            y: '0', //垂直安放位置，默认为top，可选为：'top' | 'bottom' | 'center' | {number}（y坐标，单位px）
            textStyle: {
              // 主标题文本样式
              fontFamily: "Microsoft Yahei",
              fontSize: 25,
              fontStyle: "normal",
              fontWeight: "bold",
              color: "#000",
            },
      },
      toolbox: {
          show: false,
          feature: {
              mark: {
                  show: true
              },
              dataView: {
                  show: true,
                  readOnly: false
              },
              restore: {
                  show: true
              },
              saveAsImage: {
                  show: true
              }
          }
      },
      tooltip: {
        show: true,    // 是否显示提示框组件
        trigger: 'item',    // 触发类型（'item'，数据项图形触发，主要在散点图，饼图等无类目轴的图表中使用；'axis'，坐标轴触发，主要在柱状图，折线图等会使用类目轴的图表中使用；'none'，不触发。）
        formatter: function (param){
            return param.name +': '+parseInt(param.value-number);
        },
        extraCssText: 'z-index: 9',
        showContent: true,     // 是否显示提示框浮层，默认显示
        alwaysShowContent: false,     // 是否永远显示提示框内容，默认情况下在移出可触发提示框区域后一定时间后隐藏
        triggerOn: 'mousemove|click',    // 提示框触发的条件（'mousemove'，鼠标移动时触发；'click'，鼠标点击时触发；'mousemove|click'，同时鼠标移动和点击时触发；'none'，不在 'mousemove' 或 'click' 时触发）
        confine: true,    // 是否将 tooltip 框限制在图表的区域内
        backgroundColor: 'rgba(229, 222, 222, 0.7)',    // 提示框浮层的背景颜色
        padding: 5,    // 提示框浮层内边距，单位px
        textStyle: {
            color: '#000',     // 文字的颜色
            fontStyle: 'normal',    // 文字字体的风格（'normal'，无样式；'italic'，斜体；'oblique'，倾斜字体） 
            fontWeight: 'bold',    // 文字字体的粗细（'normal'，无样式；'bold'，加粗；'bolder'，加粗的基础上再加粗；'lighter'，变细；数字定义粗细也可以，取值范围100至700）
            fontSize: '30',    // 文字字体大小
            lineHeight: '50',    // 行高 
        }
      },
      
      series: [{
          name: '数量',
          type: 'pie',
          radius: [360, 420],
          center: ['50%', '50%'],
          roseType: 'area',
          label: {
              padding: [0,0,0,0],
              normal: {
                show: true,
                position:'insideTop',
                formatter: (params) => {
                  const { name, value } = params;
                  const maxLength = 12; // 每行最多字符数
                  const maxLines = 3;  // 限制最大行数                   
                  // 把名称分割成每行最多 12 个字符
                  let wrappedName = name.match(/.{1,12}/g)?.join('\n') || name;
                  // 如果行数超过最大行数，则截断并加上省略号
                  const lines = wrappedName.split('\n');
                  if (lines.length > maxLines) {
                      wrappedName = lines.slice(0, maxLines).join('\n') + '\n...';
                  }
                  return `${wrappedName || name}`;  // 显示名称部分
              },
                  textStyle: { //标签的字体样式
                    color: '#000', //字体颜色
                    fontStyle: 'oblique',//文字字体的风格 'normal'标准 'italic'斜体 'oblique' 倾斜
                    fontWeight: 'bold',//'normal'标准'bold'粗的'bolder'更粗的'lighter'更细的或100 | 200 | 300 | 400...
                    fontFamily: 'Microsoft Yahei', //文字的字体系列
                    fontSize: 20, //字体大小
                  },
                  rotate: 'radial'
              }
          },
          labelLine: {
              normal: {
                  show: false
              }
          },
          itemStyle: {
              borderRadius: 2,
              borderWidth:6,
              borderColor:'#fff',
              color:'#A9A9A9'
          },
          data: showData,
      },
      {
        name: '数量',
        type: 'pie',
        radius: [360, 420],
        center: ['50%', '50%'],
        roseType: 'area',
        label: {
            normal: {
                show: true,
                position: 'inside', // 将值显示在饼图的底部
                formatter: function (params) {
                    const { value } = params;
                    return `${parseInt(value-number)}`; // 只显示值部分
                },
                textStyle: {
                    color: 'red',  // 控制value文本的颜色
                    fontSize: 18,   // 控制value文本的大小
                    fontWeight: 'bold',
                }
            }
        },
        labelLine: {
            normal: {
                show: false
            }
        },
        itemStyle: {
            borderRadius: 2,
            borderWidth: 6,
            borderColor: '#fff',
            color: '#A9A9A9'
        },
        data: showData,
    }
    ]
    });
}
