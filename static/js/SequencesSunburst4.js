
  var param_json = JSON.parse(document.getElementById("global_data_json").value)
  var files_path = param_json.data_path
  var journal = param_json.journals[0]
  var file = files_path + journal+'.json';
  var count_file = files_path + 'count_journal.json';

  var journal1 = journal;
  
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
  //console.log(colors1)
  // var colors1 = {
  //   "Engineering Electrical Electronic": "#65B252",
  //   "Computer Science Information Systems": "#3896ED",
  //   "Computer Science Artificial Intelligence": "#F36EA7",
  //   "Telecommunications": "#9454E6",
  //   "Sport Sciences": "#FF8800",
  //   "Chemistry Analytical": "#EB7E6A",
  //   "Multidisciplinary Sciences": "#FFD135",
  //   "PhysicsApplied": "#6A53EC"
  // };


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
      const bubble = data => d3.pack()
          .size([width1, height1])
          .padding(2)(d3.hierarchy({ children: data })
          .sum(d => d["Since 2013 Usage Count"]));
      
      const svg = d3.select('#bubble-chart')
          .attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
          .style("display", "block")
          .style("cursor", "pointer")
          .style("clip-path",`circle(50%)`)
          .style('width', width1)
          .style('height', height1)
          .on("click", (event, d) => (zoom(event, root), event.stopPropagation()));

      const root = bubble(data);
      console.log(data)
      const tooltip = d3.select('.tooltip');
      var focus = root,
          view,
          margin = 20;
      var diameter = width1;

      const node = svg.selectAll()
          .data(root.children)
          .enter().append('g')
          .attr('transform', function(d) { return "translate(" + d.x + "," + d.y + ")"; })
          .filter((d) => d)
          .sort((a, b) => a - b);
      
      const circle = node.append('circle')
          .style('fill', function(d) {
            var research_raw = d.data["Research Areas"] || "Others";
            var research_re = research_raw.replace(/&/g, "and");

            var bestMatch = null;
            var earliestIndex = Infinity;

            for (var key in colors1) {
              var index = research_re.indexOf(key);
              if (index !== -1 && index < earliestIndex) {
                earliestIndex = index;
                bestMatch = key;
              }
            }

            if (bestMatch) {
              return colors1[bestMatch];
            } else {
              return colors1["Others"] || "#999";
            }
          })
          .sort((a, b) => a - b)
          .on('mouseover', function (e, d) {
              var column_name_set = param_json.column_name_set
              d3.selectAll('.tooltip_label').each(function(da, i) {
                d3.select(this).text(column_name_set[i]+'：'+ d.data[column_name_set[i]]).attr("font-family", "Microsoft Yahei");
              });
              // console.log('#'+key)
              // tooltip.select('#company').text(d.data.name).attr("font-family", "Microsoft Yahei");
              // tooltip.select('#money').text('被引次数：'+ (d.data.money == ""?"无":d.data.money)).attr("font-family", "Microsoft Yahei");
              // tooltip.select('#ground').text('期刊名称：'+ (d.data.ground == ""?"无":d.data.ground)).attr("font-family", "Microsoft Yahei");
              // tooltip.select('#property').text('关键词：'+ (d.data.property == ""?"无":d.data.property)).attr("font-family", "Microsoft Yahei");
              // tooltip.select('#address').text('发表单位：'+ (d.data.address == ""?"无":d.data.address)).attr("font-family", "Microsoft Yahei");
              tooltip.style('visibility', 'visible').style('z-index', 5);

              d3.select(this).style('stroke', '#222');
          })
          .on('mousemove', function(e) {
            var mousePos = mousePosition(e);
            var  xOffset = 20;
            var  yOffset = 25;
            tooltip.style('top', `${(mousePos.y - yOffset)}px`)
                    .style('left', `${(mousePos.x + xOffset)}px`);})
          // .on('mousemove', e => tooltip.attr('transform', `translate(${width1 / 2}, ${height1 / 2})`))
          .on('mouseout', function () {
              d3.select(this).style('stroke', 'none');
              return tooltip.style('visibility', 'hidden');
          })
          .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));
      
      const label = node.append('text')
          .attr('dy', 2)
          .style("display", "none")
          .attr("font-size",60)
          .attr("font-family", "Microsoft Yahei")
          .text(d => d.data["Article Title"].substring(0, d.r));
      
      circle.transition()
          .ease(d3.easeExpInOut)
          .duration(1000)
          .attr('r', function(d){console.log(data.length);if(data.length==1){return 100;}else{return d.r}});
      
      label.transition()
          .delay(700)
          .ease(d3.easeExpInOut)
          .duration(1000)
          .style('opacity', 1);
      
      zoomTo([root.x, root.y, root.r * 2]);

      function zoom(event, d) {
        var focus0 = focus; focus = d;
        var transition = d3.transition()
            .duration(event.altKey ? 7500 : 750)
            .tween("zoom", d => {
            const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
            return t => zoomTo(i(t));
            });
        label
            .style("fill-opacity", function(d) { return root !== focus ? 1 : 0; })
            .style("display", function(d) { return "inline" || (focus !== root); });
            // .on("start", function(d) {if (root === focus) this.style.display = "inline"; })
            // .on("end", function(d) {if (root !== focus) this.style.display = "none";});
        return svg.node();
      };

      function zoomTo(v) {
          var k = diameter / v[2]; view = v;
          console.log(diameter);
          console.log(v);
          node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
        //   label.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
          circle.attr("r", function(d) { return d.r * k; });
      }

  };

  function generateChart1(data) {
    const bubble = data => d3.pack()
        .size([width1, height1])
        .padding(2)(d3.hierarchy({ children: data })
        .sum(d => d["Since 2013 Usage Count"]))
        .sort(function (a, b) {
          const mod = order === "desc" ? -1 : 1;
          return mod * (a.value - b.value);
        });

    const root1 = bubble(data);
    const tooltip = d3.select('.tooltip');
    var focus = root1,
          view,
          margin = 20;

    const svg = d3.select('#bubble-chart1')
        .attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
        .style("display", "block")
        .style("cursor", "pointer")
        .style("clip-path",`circle(50%)`)
        .style('width', width1)
        .style('height', height1)
        .on("click", (event, d) => (zoom(event, root1), event.stopPropagation()));
    
    var diameter = width1;

    const node = svg.selectAll()
        .data(root1.children)
        .enter().append('g')
        .attr('transform', `translate(${width1 / 2}, ${height1 / 2})`);
    
     
    
    const circle = node.append('circle')
        .style('fill', function(d) {
          var research_raw = d.data["Research Areas"] || "Others";
          var research_re = research_raw.replace(/&/g, "and");

          var bestMatch = null;
          var earliestIndex = Infinity;

          for (var key in colors1) {
            var index = research_re.indexOf(key);
            if (index !== -1 && index < earliestIndex) {
              earliestIndex = index;
              bestMatch = key;
            }
          }

          if (bestMatch) {
            return colors1[bestMatch];
          } else {
            return colors1["Others"] || "#999";
          }
        })
        .on('mouseover', function (e, d) {  
            var column_name_set = param_json.column_name_set
            d3.selectAll('.tooltip_label').each(function(da, i) {
              d3.select(this).text(column_name_set[i]+'：'+ d.data[column_name_set[i]]).attr("font-family", "Microsoft Yahei");
            });

            // tooltip.select('#company').text(d.data.name);
            // tooltip.select('#money').text('被引次数：'+ (d.data.money == ""?"无":d.data.money)).attr("font-family", "Microsoft Yahei");
            //   tooltip.select('#ground').text('期刊名称：'+ (d.data.ground == ""?"无":d.data.ground)).attr("font-family", "Microsoft Yahei");
            //   tooltip.select('#property').text('关键词：'+ (d.data.property == ""?"无":d.data.property)).attr("font-family", "Microsoft Yahei");
            //   tooltip.select('#address').text('发表单位：'+ (d.data.address == ""?"无":d.data.address)).attr("font-family", "Microsoft Yahei");
            tooltip.style('visibility', 'visible');

            d3.select(this).style('stroke', '#222');
        })
        .on('mousemove', function(e) {
          var mousePos = mousePosition(e);
          var  xOffset = 20;
          var  yOffset = 25;
          tooltip.style('top', `${(mousePos.y - yOffset)}px`)
                                    .style('left', `${(mousePos.x + xOffset)}px`);})
        // .on('mousemove', e => tooltip.attr('transform', `translate(${width1 / 2}, ${height1 / 2})`))
        .on('mouseout', function () {
            d3.select(this).style('stroke', 'none');
            return tooltip.style('visibility', 'hidden');
        })
        .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

    const label = node.append('text')
        .attr('dy', 2)
        .style("display", "none")
        .attr("font-size",60)
        .attr("font-family", "Microsoft Yahei")
        .text(d => d.data["Article Title"].substring(0, d.r));

    circle.transition()
        .ease(d3.easeExpInOut)
        .duration(1000)
        .attr('r', function(d){console.log(data.length);if(data.length==1){return 100;}else{return d.r}});
    
    label.transition()
        .delay(700)
        .ease(d3.easeExpInOut)
        .duration(1000)
        .style('opacity', 1)

    zoomTo([root1.x, root1.y, root1.r * 2]);

    function zoom(event, d) {
      var focus0 = focus; focus = d;
      var transition = d3.transition()
          .duration(event.altKey ? 7500 : 750)
          .tween("zoom", d => {
          const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
          return t => zoomTo(i(t));
          });
      label
          .style("fill-opacity", function(d) { return root1 !== focus ? 1 : 0; })
          .style("display", function(d) { return "inline" || (focus !== root1); });
          // .on("start", function(d) {if (root1 === focus) this.style.display = "none"; })
          // .on("end", function(d) {if (root1 !== focus) this.style.display = "inline";});
      return svg.node();
    };

    function zoomTo(v) {
        var k = diameter / v[2]; view = v;
        console.log(diameter);
        console.log(v);
        node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
      //   label.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
        circle.attr("r", function(d) { return d.r * k; });
    }
  };

  (async () => {
      data = await d3.json(file).then(data => data);
      generateChart(data);
  })();

  // var dataset = [1,1,1,1,1,1,1,1];

  // var dataset_name = ["Engineering Electrical Electronic","Computer Science Information Systems","Computer Science Artificial Intelligence","Telecommunications","Sport Sciences","Chemistry Analytical","Multidisciplinary Sciences","PhysicsApplied"];

  // let colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'];
  // let colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'];
// var colors = ["#66CCCC","#FF99CC","#FF9999","#99CC66","#5151A2","#FF9900","#FF6600","#CCFF66"];

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

    var  emblemText= ['J O U R N A L', 'J O U R N A L', 'J O U R N A L'];    
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
          file = files_path+e.data.name+'.json'; 
          journal = e.data.name;
          var names1 = []; //类别数组（用于存放饼图的类别）
          var brower1 = [];
          $.ajax({
            url: files_path+"count_journal.json",
            data: {},
            type: 'GET',
            success: function(data) {
                //请求成功时执行该函数内容，result即为服务器返回的json对象
                $.each(data, function(index, item) {
                  if (param_json.journals.includes(item.journal)) {
                    names1.push(item.value); //挨个取出类别并填入类别数组
                    brower1.push({
                        name: item.journal,
                        value: item.infected
                    });
                  }
                });
                hrFun(brower1);
            },
          });
          (async () => {
            var data = await d3.json(file).then(data => data);
            generateChart(data);
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
      var file1 = files_path+e.data.name+'.json'; 
      journal1 = e.data.name
      var names1 = []; //类别数组（用于存放饼图的类别）
      var brower1 = [];
      $.ajax({
        url: files_path+"count_journal.json",
        data: {},
        type: 'GET',
        success: function(data) {
            //请求成功时执行该函数内容，result即为服务器返回的json对象
            $.each(data, function(index, item) {
              if (param_json.journals.includes(item.journal)) {
                names1.push(item.value); //挨个取出类别并填入类别数组
                brower1.push({
                    name: item.journal,
                    value: item.infected
                });
              }
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
          var data = await d3.json(file1).then(data => data);
          generateChart1(data);
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
        file = files_path+e.data.name+'.json'; 
        journal1 = e.data.name;
        var names1 = []; //类别数组（用于存放饼图的类别）
        var brower1 = [];
        $.ajax({
          url: files_path+"count_journal.json",
          data: {},
          type: 'GET',
          success: function(data) {
              //请求成功时执行该函数内容，result即为服务器返回的json对象
              $.each(data, function(index, item) {
                if (param_json.journals.includes(item.journal)) {
                  names1.push(item.value); //挨个取出类别并填入类别数组
                  brower1.push({
                      name: item.journal,
                      value: item.infected
                  });
                }
              });
              hrFun2(brower1);
          },
        });
        (async () => {
          var data = await d3.json(file).then(data => data);
          generateChart1(data);
        })();
        // 单击事件的代码执行区域
        // ...
    }, time)
}
function double1 (e) {
    clearTimeout(timeOut); // 清除第二个单击事件
    journal1 = e.data.name
    var file1 = files_path+e.data.name+'.json'; 
    var names1 = []; //类别数组（用于存放饼图的类别）
    var brower1 = [];
    const svg = d3.select('#bubble-chart');
    svg.selectAll("g").remove()
    .transition()
    .delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
    .duration(secDur);
    $.ajax({
      url: files_path+"count_journal.json",
      data: {},
      type: 'GET',
      success: function(data) {
          //请求成功时执行该函数内容，result即为服务器返回的json对象
          $.each(data, function(index, item) {
            if (param_json.journals.includes(item.journal)) {
              names1.push(item.value); //挨个取出类别并填入类别数组
              brower1.push({
                  name: item.journal,
                  value: item.infected
              });
            }
          });
          hrFun(brower1);
      },
    });
    (async () => {
      var data = await d3.json(file1).then(data => data);
      generateChart(data);
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
      file = files_path+journal+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'.json'; 
      (async () => {
        var data = await d3.json(file).then(data => data);
        generateChart(data);
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
  var file1 = files_path+journal+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'.json'; 
  var names1 = []; //类别数组（用于存放饼图的类别）
  var brower1 = [];
  $.ajax({
    url: files_path+"count_journal.json",
    data: {},
    type: 'GET',
    success: function(data) {
        //请求成功时执行该函数内容，result即为服务器返回的json对象
        $.each(data, function(index, item) {
          if (param_json.journals.includes(item.journal)) {
            names1.push(item.value); //挨个取出类别并填入类别数组
            brower1.push({
                name: item.journal,
                value: item.infected
            });
          }
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
      var data = await d3.json(file1).then(data => data);
      generateChart1(data);
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
    file = files_path+journal1+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'.json'; 
    (async () => {
      var data = await d3.json(file).then(data => data);
      generateChart1(data);
    })();
    // 单击事件的代码执行区域
    // ...
}, time)
}
function piedouble1 (e) {
clearTimeout(timeOut); // 清除第二个单击事件
var file1 = files_path+journal1+'-'+e.srcElement.nextSibling.lastChild.innerHTML+'.json'; 
var names1 = []; //类别数组（用于存放饼图的类别）
var brower1 = [];
const svg = d3.select('#bubble-chart');
svg.selectAll("g").remove()
.transition()
.delay(function (d, i) {return arcAnimDur + i * secIndividualdelay;})
.duration(secDur);
$.ajax({
  url: files_path+"count_journal.json",
  data: {},
  type: 'GET',
  success: function(data) {
      //请求成功时执行该函数内容，result即为服务器返回的json对象
      $.each(data, function(index, item) {
        if (param_json.journals.includes(item.journal)) {
          names1.push(item.value); //挨个取出类别并填入类别数组
          brower1.push({
              name: item.journal,
              value: item.infected
          });
        }
      });
      hrFun1(brower1);
  },
});
(async () => {
  var data = await d3.json(file1).then(data => data);
  generateChart(data);
  draw();
})();
// 双击的代码执行区域
// ...
}

  var draw1 = function draw1() {
    
    var  emblemText=  ['J O U R N A L', 'J O U R N A L', 'J O U R N A L'];   
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
  
    // const dataq = [
      
    //   {value: 1, text: "Computer Science Information Systems", color: '#666666'},
    //   {value: 1, text: "Computer Science Artificial Intelligence", color: '#006634'},
    //   {value: 1, text: "Telecommunications", color: '#66999A'},
    //   {value: 1, text: "Sport Sciences", color: '#FEE100'},
    //   {value: 1, text: "Chemistry Analytical", color: '#FF7F00'},
    //   {value: 1, text: "Multidisciplinary Sciences", color: '#6599FF'},
    //   {value: 1, text: "PhysicsApplied", color: '#999999'},
    //   {value: 1, text: "Engineering Electrical Electronic", color: '#99CCCD'}];
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
      url: files_path+"count_journal.json",
      data: {},
      type: 'GET',
      success: function(data) {
          //请求成功时执行该函数内容，result即为服务器返回的json对象
          $.each(data, function(index, item) {
            if (param_json.journals.includes(item.journal)) {
              names.push(item.value); //挨个取出类别并填入类别数组
              brower.push({
                  name: item.journal,
                  value: item.infected
              });
            }
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
      var textdata = journal;
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
    var textdata = journal1;
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
