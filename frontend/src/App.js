import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const [emotion, setEmotion] = useState("");
  const [loading, setLoading] = useState(false);

  const capture = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    const base64Image = imageSrc.split(",")[1];

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", {
        image: base64Image,
      });
      setEmotion(res.data.emotion);
    } catch (error) {
      setEmotion("Error detecting emotion");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Real-Time Emotion Detector</h1>
      <div className="card">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="webcam"
        />
        <button onClick={capture}>Detect Emotion</button>
        {loading && <p>Analyzing...</p>}
        {emotion && !loading && <p className="emotion">{emotion}</p>}
      </div>
    </div>
  );
}

export default App;
