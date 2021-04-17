import React from "react";
import ReactDOM from 'react-dom';

class Main extends React.Component { 
    constructor(props){
        super(props);
        this.state = {
            data: {},
            sampleImages: [],
            preferences: [],
            sliderValue: 0
        }
        this.handleLeftClick = this.handleLeftClick.bind(this);
        this.handleRightClick = this.handleRightClick.bind(this);
        this.submitPreferences = this.submitPreferences.bind(this);
        this.submitDefaultPreferences = this.submitDefaultPreferences.bind(this);
        this.setInitialPreferences = this.setInitialPreferences.bind(this);
        this.getSegments = this.getSegments.bind(this);
        this.set_slider_value = this.set_slider_value.bind(this);
        this.setImages = this.setImages.bind(this)   
        this.set_black_images = this.set_black_images.bind(this);
        this.reset_slider = this.reset_slider.bind(this);  
        this.reset_dashboard = this.reset_dashboard.bind(this); 
    }
    componentDidMount() {
        this.reset_dashboard();
    }

    set_black_images() {
        console.log("Setting Images");
        var img_tags = document.getElementsByTagName('img');

        for (var i=0; i< img_tags.length; i++)
        {
            img_tags[i].src = "/static/black_image.jpeg";
        }
    }

    getSegments() {
        fetch("/get_segments",{
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        })
        .then(response => response.json())
        .then(data => {
            // Sample Imges URL to Images array
            let images = [];
            let defaultName = "sample_"
            for(let i=1;i<9;i++) {
                let obj_key = defaultName + i;
                images.push(data[obj_key])
            }
            this.setState({
                data: {...data},
                sampleImages: [...images],
                sliderValue: data.level
            })
        })
    }

    handleLeftClick(e,id, organ) {
        let temp_preferences = this.state.preferences;
        temp_preferences[id][organ] = '1';
        this.setState({
            preferences: temp_preferences
        })

    }

    handleRightClick(e, id, organ) {
        e.preventDefault()
        let temp_preferences = this.state.preferences;
        temp_preferences[id][organ] = '0';
        this.setState({
            preferences: temp_preferences
        })
    }

    submitDefaultPreferences() {
        let temp_samples = [1,2,3,4,5,6,7,8]
            let temp_preferences = []
            let temp_organ_preference = {
                "Accept/Reject": null,
                "Torso": "1",
                "Left Lung": '0',
                "Right Lung": '1',
                "Spine": '1',
                "Heart": '0'
            }
            for(let i=1; i<=temp_samples.length; i++)  {
                let id = i.toString();
                if(i%2 === 0) temp_preferences.push({...temp_organ_preference, "Accept/Reject":'1'})
                else temp_preferences.push({...temp_organ_preference, "Accept/Reject":'0'})
            }
            this.setState({
                preferences: [...temp_preferences]
            })
    }

    setInitialPreferences() {
        console.log("Initial Preferences");
        let temp_samples = [1,2,3,4,5,6,7,8]
        let temp_preferences = []
        let temp_organ_preference = {
            "Accept/Reject": null,
            "Torso": null,
            "Left Lung": null,
            "Right Lung": null,
            "Spine": null,
            "Heart": null
        }
        for(let i=1; i<=temp_samples.length; i++)  {
            let id = i.toString();
            temp_preferences.push({...temp_organ_preference})
        }
        this.setState({
            preferences: [...temp_preferences]
        })
    }

    submitPreferences() {
        let final_preferences = this.state.preferences;
        let complete_Flag = true;
        for(let i=0; i<final_preferences.length; i++) {
            let pref = final_preferences[i];
            for(let key in pref){
                if(pref[key] === null) {
                    complete_Flag = false
                    break;
                }
            }
            if(!complete_Flag) break;
        }
        if(complete_Flag){
            let pref_req = {}
            for(let i=0;i<final_preferences.length;i++){
                pref_req[i+1] = {...final_preferences[i]}
            }
            $.ajax({
                type:"POST",
                url: "/submit_prefs",
                dataType:'json',
                data: {preferences: JSON.stringify(pref_req)},
                success: function(data){
                   alert("All the preferences have been submitted")
                }
            }).then(()=>{
                this.reset_dashboard();
            })
        }
        else alert("Please submit the preferences for all the samples")
    }

    set_slider_value(){
        document.getElementById("pca_slider").value = this.state.sliderValue;
    }

    reset_slider(){
        this.setState({
            sliderValue: 0
        },()=> {
            this.set_slider_value()
        })
    }

    setImages() {
        document.getElementById("source").src = this.state.data.img1
        document.getElementById("dest").src = this.state.data.img2
        document.getElementById("interpolated").src = this.state.data.img_mid
    }

    reset_dashboard(){
        this.reset_slider();
        this.setInitialPreferences();
        this.setState({
            data: {}
        })
        this.set_black_images();
        this.getSegments()
    }

    render() { 
        this.set_slider_value();
        if(Object.keys(this.state.data).length !== 0) this.setImages()
        let temp_samples = [1,2,3,4,5,6,7,8]
        const render_images = temp_samples.map((img_src,index) => {
                // Update it to original images received (this.state.sampleImages)
                return (
                    <div key={index+1} >
                        {
                            this.state.preferences.length ?
                            <div className="sample-image-container">
                                <img 
                                    src= {"/static/black_image.jpeg"}   // update this to img_src once API is fixed
                                    className={ this.state.preferences[index]["Accept/Reject"] === null
                                                        ? "sample-image"
                                                        :  this.state.preferences[index]["Accept/Reject"] === "1" 
                                                        ? "sample-image sample-image-border-green"
                                                        : "sample-image sample-image-border-blue" }
                                />

                                <div>
                                    <div 
                                        className={ this.state.preferences[index]["Torso"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Torso"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" } 
                                        onClick={(e=event,sample_id = index,organ="Torso") => this.handleLeftClick(e,sample_id,organ)} 
                                        onContextMenu={(e=event,sample_id = index,organ="Torso") => this.handleRightClick(e, sample_id, organ)}
                                    >
                                        Torso
                                    </div>
                                    <div 
                                        className={ this.state.preferences[index]["Left Lung"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Left Lung"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" }
                                        onClick={(e=event,sample_id = index,organ="Left Lung") => this.handleLeftClick(e,sample_id,organ)} 
                                        onContextMenu={(e=event,sample_id = index,organ="Left Lung")=> this.handleRightClick(e, sample_id, organ)}
                                    > 
                                        Lung (L)
                                    </div>
                                    <div 
                                    className={ this.state.preferences[index]["Right Lung"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Right Lung"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" }
                                    onClick={(e=event,sample_id = index,organ="Right Lung")=> this.handleLeftClick(e,sample_id,organ)}
                                    onContextMenu={(e=event,sample_id = index,organ="Right Lung")=> this.handleRightClick(e, sample_id, organ)}
                                    >
                                        Lung (R)
                                    </div>
                                    <div 
                                    className={ this.state.preferences[index]["Spine"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Spine"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" }
                                    onClick={(e=event,sample_id = index,organ="Spine") => this.handleLeftClick(e,sample_id,organ)} 
                                    onContextMenu={(e=event,sample_id = index,organ="Spine")=> this.handleRightClick(e, sample_id, organ)} 
                                    >
                                        Spine
                                    </div>
                                    <div 
                                    className={ this.state.preferences[index]["Heart"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Heart"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" }
                                    onClick={(e=event,sample_id = index,organ="Heart") => this.handleLeftClick(e,sample_id,organ)} 
                                    onContextMenu={(e=event,sample_id = index,organ="Heart")=> this.handleRightClick(e, sample_id, organ)} 
                                    >
                                        Heart
                                    </div>
                                    <div 
                                    className={ this.state.preferences[index]["Accept/Reject"] === null
                                                        ? "organ-description"
                                                        :  this.state.preferences[index]["Accept/Reject"] === "1" 
                                                        ? "organ-description organ-description-green"
                                                        : "organ-description organ-description-red" }
                                    onClick={(e=event,sample_id = index,organ="Accept/Reject") => this.handleLeftClick(e,sample_id,organ)} 
                                    onContextMenu={(e=event,sample_id = index,organ="Accept/Reject")=> this.handleRightClick(e, sample_id, organ)} 
                                    >
                                        Accept / Reject
                                    </div>
                                </div>
                            </div>
                            : ""
                        }
                    </div>
                )
            })
            
        return (
            <div className="sample-neighborhood-container">
                <div className="render-samples-container">
                    {render_images}
                </div>
                <div className="buttons-container">
                    <button className="submit-preferences-button" onClick = {()=> this.submitDefaultPreferences()}>Default Preferences</button>
                    <button className="submit-preferences-button" onClick = {()=> this.setInitialPreferences()}>Reset Preferences</button>
                    <button className="submit-preferences-button" onClick = {()=> this.submitPreferences()}>Submit Preferences</button>
                </div>
            </div>
        );
    }    
}

export default Main;