import * as tfvis from "@tensorflow/tfjs-vis";
const statusElement = document.getElementById("status");
const messageElement = document.getElementById("message");
const imagesElement = document.getElementById("images");
const visualiseElement = document.getElementById("log");
const Layer0Element = document.getElementById("Layer0");
const Layer1Element = document.getElementById("Layer1");
const Layer2Element = document.getElementById("Layer2");
const Layer3Element = document.getElementById("Layer3");
const Layer4Element = document.getElementById("Layer4");
const Layer5Element = document.getElementById("Layer5");
const Layer6Element = document.getElementById("Layer6");
const Layer7Element = document.getElementById("Layer7");

export function logStatus(message) {
  statusElement.innerText = message;
}

export function logVisualise(message) {
  visualiseElement.innerText = message;
}
export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(batch, predictions, labels) {
  const testExamples = batch.xs.shape[0];
  imagesElement.innerHTML = "";
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
    const div = document.createElement("div");
    div.className = "pred-container";

    const canvas = document.createElement("canvas");
    canvas.className = "prediction-canvas";
    draw(image.flatten(), canvas);

    const pred = document.createElement("div");

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${correct ? "pred-correct" : "pred-incorrect"}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

export function showLayer(output, j) {
  const numfilters = output.shape[3];
  const size = output.shape[1];
  const div = document.createElement("div");
  div.className = "row";
  Layer0Element.innerHTML = "";
  Layer1Element.innerHTML = "";
  Layer2Element.innerHTML = "";
  Layer3Element.innerHTML = "";
  Layer4Element.innerHTML = "";
  Layer5Element.innerHTML = "";
  Layer6Element.innerHTML = "";
  Layer7Element.innerHTML = "";
  for (let i = 0; i < numfilters; i++) {
    const image = output.slice([0, 0, 0, i], [1, size, size, 1]);
    const div1 = document.createElement("div");
    div1.className = "column";
    const canvas = document.createElement("canvas");
    canvas.className = "layer1-canvas";
    canvas.style.marginRight = "0.5em";
    canvas.style.marginTop = "1em";
    show(image.flatten(), canvas, size);
    div.appendChild(canvas);
    switch (j) {
      case 0:
        Layer0Element.appendChild(div);
        dLayer0.style.display = "none";
        break;
      case 1:
        Layer1Element.appendChild(div);
        document.getElementById("Layer1").style.display = "none";
        break;
      case 2:
        Layer2Element.appendChild(div);
        document.getElementById("Layer2").style.display = "none";
        break;
      case 3:
        Layer3Element.appendChild(div);
        document.getElementById("Layer3").style.display = "none";
        break;
      case 4:
        Layer4Element.appendChild(div);
        document.getElementById("Layer4").style.display = "none";
        break;
      case 5:
        Layer5Element.appendChild(div);
        document.getElementById("Layer5").style.display = "none";
        break;
      case 6:
        Layer6Element.appendChild(div);
        document.getElementById("Layer6").style.display = "none";
        break;
      case 7:
        Layer7Element.appendChild(div);
        document.getElementById("Layer7").style.display = "none";
        break;
      default:
        console.log("no such element, lol");
    }
  }
}

const lossLabelElement = document.getElementById("loss-label");
const accuracyLabelElement = document.getElementById("accuracy-label");
const lossValues = [[], []];
export function plotLoss(batch, loss, set) {
  const series = set === "train" ? 0 : 1;
  lossValues[series].push({ x: batch, y: loss });
  const lossContainer = document.getElementById("loss-canvas");
  tfvis.render.linechart(
    lossContainer,
    { values: lossValues, series: ["train", "validation"] },
    {
      xLabel: "Batch #",
      yLabel: "Loss",
      width: 400,
      height: 300,
    }
  );
  lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
}

const accuracyValues = [[], []];
export function plotAccuracy(batch, accuracy, set) {
  const accuracyContainer = document.getElementById("accuracy-canvas");
  const series = set === "train" ? 0 : 1;
  accuracyValues[series].push({ x: batch, y: accuracy });
  tfvis.render.linechart(
    accuracyContainer,
    { values: accuracyValues, series: ["train", "validation"] },
    {
      xLabel: "Batch #",
      yLabel: "Accuracy",
      width: 400,
      height: 300,
    }
  );
  accuracyLabelElement.innerText = `last accuracy: ${(accuracy * 100).toFixed(
    1
  )}%`;
}

export function show(image, canvas, size) {
  const [width, height] = [size, size];
  canvas.width = 4.5 * width;
  canvas.height = 4.5 * height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  ctx.drawImage(canvas, 0, 0, 4 * canvas.width, 4 * canvas.height);
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function getTrainEpochs() {
  return Number.parseInt(document.getElementById("train-epochs").value);
}

export function getLearningRate() {
  return Number.parseFloat(document.getElementById("learning-rate").value);
}

export function getBatchSize() {
  return Number.parseInt(document.getElementById("batch-size").value);
}

export function getOptimizer() {
  return document.getElementById("optimizer").value;
}

export function setTrainButtonCallback(callback) {
  const trainButton = document.getElementById("train");
  trainButton.addEventListener("click", () => {
    trainButton.setAttribute("disabled", true);
    callback();
  });
}

export function setVisualiseButton0Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer0");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}

export function setVisualiseButton1Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer1");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton2Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer2");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton3Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer3");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton4Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer4");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton5Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer5");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton6Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer6");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
export function setVisualiseButton7Callback(callback) {
  const visualiseButton = document.getElementById("visualise-layer7");
  visualiseButton.addEventListener("click", () => {
    visualiseButton.setAttribute("disabled", true);
    callback();
  });
}
