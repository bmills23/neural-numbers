async function loadModel() {
    try {
        const model = await tf.loadLayersModel('./model/model.json');

        console.log('Model loaded successfully!');

        // Get the canvas element
        const canvas = document.getElementById('canvas');

        // Get the 2D rendering context
        const ctx = canvas.getContext('2d');

        // Add an event listener to the canvas for mouse/mobile movement
        let isDrawing = false;

        if (navigator.userAgent.indexOf('Mobile') == false) {
            canvas.addEventListener('touchstart', function(event) {
                isDrawing = true;
                draw(event);
            });
            canvas.addEventListener('touchmove', function(event) {
                if (isDrawing) {
                    draw(event);
                }
            });
            canvas.addEventListener('touchend', function() {
                isDrawing = false;
            });
        } else {
            canvas.addEventListener('mousedown', function(event) {
                isDrawing = true;
                draw(event);
            });
            canvas.addEventListener('mousemove', function(event) {
                if (isDrawing) {
                    draw(event);
                }
            });
            canvas.addEventListener('mouseup', function() {
                isDrawing = false;
            });
        }
    
        // Function to draw on the canvas
        function draw(event) {
            // Get the mouse coordinates
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Draw a small rectangle at the current mouse position
            ctx.fillStyle = 'black';
            ctx.fillRect(x, y, 5, 5);
        }

        // Function to predict the number based on the drawn image
        async function predict() {
            // Preprocess the image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const preprocessedData = preprocess(imageData.data);
            console.log(preprocessedData);
            // Reshape the data to match the input shape of the model
            const reshapedData = tf.reshape(preprocessedData, [1, 28, 28, 1]);
            // Normalize the data
            const normalizedData = reshapedData.div(255);
            // Make the prediction
            const prediction = model.predict(normalizedData);
            prediction.print();
            // Get the predicted number
            const predictedNumber = tf.argMax(prediction, axis=1).dataSync()[0];
            console.log(imageData, reshapedData, normalizedData, prediction, predictedNumber);
        }

        function preprocess(imageData) {
            // Convert the image data to a tensor
            const tensor = tf.browser.fromPixels(imageData);
            console.log(tensor);
            // Convert the tensor to grayscale
            const grayscale = tensor.mean(2);
            console.log(grayscale);
            // Add an extra dimension to the grayscale tensor
            const expanded = grayscale.expandDims(2);
            console.log(expanded);
            // Resize the grayscale image to match the input shape of the model
            const resized = tf.image.resizeBilinear(expanded, [28, 28]);
            console.log(resized);
            // Return the preprocessed data
            return resized;
        }

        // Add an event listener to the predict button
        const predictButton = document.getElementById('predict-button');
        predictButton.addEventListener('click', predict);

        const clearButton = document.getElementById('clear-button');
        clearButton.addEventListener('click', function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });
        
    }

    catch (err) {
        console.log('Model not found in local storage!');
    }    
    
}

loadModel();
