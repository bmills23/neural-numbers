async function loadModel() {
    const model = await tf.loadLayersModel('./model/model.json');
    console.log('Model loaded successfully!');
    
    console.log(model.summary());
}

loadModel();