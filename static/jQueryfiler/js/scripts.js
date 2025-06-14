function ExecCallbackJsonData(data){
	if(!data) return;
	if(!data.authenticity_token) return;
	if(!data.data) return;
	if(data.data.script){
		eval(data.data.script);	
	}
}

function htmlEntities(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}