function render_source_dest(src_path, dest_path)
{
	console.log(src_path);
	console.log(dest_path);
	document.getElementById("source").src = src_path + "?" + new Date().getTime();
	document.getElementById("dest").src = dest_path + "?" + new Date().getTime();
}

function generate()
{
	$.ajax({
		type:"POST",
		url: "/generate_images",
		success: function(data){
			render_source_dest(data['source'], data['dest']);
		    render_interp(data['interpolated']);
		}
	})
}

function render_interp(int_path)
{
	console.log(int_path);
	document.getElementById("interpolated").src = int_path + "?" + new Date().getTime();
}

var slider = document.getElementById("slider");
slider.oninput = function(){

	$.ajax({

		type:"POST",
		url: "/interpolate",
		dataType:'json',
		data: {inc: this.value/100},
		success: function(data){
			render_interp(data['interpolated']);
		}
	})
}
