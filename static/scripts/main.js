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
    document.getElementById("interpolated").src = int_path + "?" + new Date().getTime();
}

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