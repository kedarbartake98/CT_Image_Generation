// ############# Global Variables ##############
var pref_dict = {};

var sample_mapping = {
	'sample_1' : 1,
	'sample_2' : 2,
	'sample_3' : 3,
	'sample_4' : 4,
	'sample_5' : 5,
	'sample_6' : 6,
	'sample_7' : 7,
	'sample_8' : 8
};

var colors = {
	blue: "#0275d8",
	green: "#5cb85c"
};

var curr_sample = 0;
var samples = Object.keys(sample_mapping);

var curr_pref = {};

// ############# Utility functions ###############

function set_black_images()
{
	var img_tags = document.getElementsByTagName('img');

	for (var i=0; i< img_tags.length; i++)
	{
		img_tags[i].src = "static/black_image.jpeg";
	}
}

function reset_prefs()
{
	$('input[type="checkbox"]:checked').prop('checked',false);
}

function clear_pref_dict()
{
	pref_dict = {};
}

function reset_slider()
{
	document.getElementById('pca_slider').value = 0;
}

function highlight_sample(img_tag_id, color)
{
	document.getElementById(img_tag_id).style.border="5px solid "+color;
}

function remove_highlight(img_tag_id)
{
	document.getElementById(img_tag_id).style.border="";
}

// #############################################################################

window.onload = function()
{
	set_black_images();
	reset_prefs();
	clear_pref_dict();
	reset_slider();
	highlight_sample('sample_1', colors['blue']);
}

// function check_curr_pref()
// {
// 	pref_length = Object.keys(curr_pref).length;

// 	if (pref_length < 6 || !(curr_sample in pref_dict)) 
// 	{
// 		// console.log("false triggered");
// 		return false;
// 	}

// 	else if (pref_length == 6)
// 	{
// 		if (curr_sample in pref_dict)
// 		{
// 			var set_pref = prompt("Do you want to submit this as a new preference\
// 			for the highlighted frame", "Yes");

// 			if (set_pref!=null)
// 			{
// 				submit_pref();
// 			}

// 			else
// 			{
// 				//pass
// 			}
// 		}

// 		else
// 		{
// 			var set_pref = prompt("Do you want to submit this as a preference\
// 			for the highlighted frame", "Yes");
// 			if (set_pref!=null)
// 			{
// 				submit_pref();
// 			}

// 			else
// 			{
// 				//pass
// 			}
// 		}
		
// 	}

// 	else
// 	{
// 		highlight_sample(sample_mapping[curr_sample], colors['green']);
// 		return true;
// 	}
// }

function select_prev_sample()
{
	// if (check_curr_pref()==false)
	// {
	// 	alert("Please select all preferences for current sample and submit");
	// }

	if (curr_sample>0)
	{
		if (!((curr_sample+1) in pref_dict))
		{
			remove_highlight(samples[curr_sample]);
		}

		curr_sample -= 1;
		highlight_sample(samples[curr_sample], colors['blue']);
	}

}

function select_next_sample()
{
	// while(check_curr_pref()==false)
	// {
	// 	// pass
	// }
	console.log('next sample selection ..');
	if (curr_sample<7)
	{
		if (!((curr_sample+1) in pref_dict))
		{
			remove_highlight(samples[curr_sample]);
		}

		curr_sample += 1;
		highlight_sample(samples[curr_sample], colors['blue']);
	}	
}

function submit_pref()
{
	pref_dict[curr_sample] = curr_pref;
	highlight_sample(samples[curr_sample], colors['green']);
}
// #############################################################################

// Implement the slider increment decrement functionality
 
function increment_slider()
{
	var curr_val = document.getElementById('pca_slider').value;
	// console.log(curr_val);
	var new_val = parseInt(curr_val)+1
	console.log(new_val);

	if(new_val<=100)
	{
		document.getElementById('pca_slider').value = new_val;
		// call modified render function
		interpolate(new_val);
	}
}

function decrement_slider()
{
	var curr_val = document.getElementById('pca_slider').value;
	// console.log(curr_val);
	var new_val = parseInt(curr_val)-1
	// console.log(new_val);

	if(new_val>=0)
	{
		document.getElementById('pca_slider').value = new_val;
		// call modified render function
		interpolate(new_val);
	}
}

function slider_response(slider_value)
{
	var slider_inp = parseInt(slider_value);
	interpolate(slider_inp);
}

function render_source_dest(src_path, dest_path)
{
	console.log(src_path);
	console.log(dest_path);
	document.getElementById("source").src = src_path + "?" + new Date().getTime();
	document.getElementById("dest").src = dest_path + "?" + new Date().getTime();

	// Get data from RL algo

	// var ids = ['image_01', 'image_02', 'image_03', 'image_04',
	//         'image_11', 'image_12', 'image_13', 'image_14'];

	// var x;

	// for (x of ids)
	// {
	//  console.log(x);
	//  document.getElementById(x).src = dest_path + "?" + new Date().getTime();
	// }
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
	// console.log(int_path);
	document.getElementById("interpolated").src = int_path + "?" + new Date().getTime();
}

// var slider = document.getElementById("slider");
// slider.oninput = function(){

//  $.ajax({

//      type:"POST",
//      url: "/interpolate",
//      dataType:'json',
//      data: {inc: this.value/100},
//      success: function(data){
//          render_interp(data['interpolated']);
//      }
//  })
// }

function interpolate(slider_val)
{
	$.ajax({

		type:"POST",
		url: "/interpolate",
		dataType:'json',
		data: {inc: slider_val/100},
		success: function(data){
			render_interp(data['interpolated']);
		}
	})
}
