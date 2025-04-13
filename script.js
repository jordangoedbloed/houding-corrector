const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const feedbackElement = document.getElementById('feedback');

// Configuratie
const config = {
    sensitivity: 20, // Standaard gevoeligheid (graden)
    minSamples: 20   // Minimale trainingsvoorbeelden
};

// Globale variabelen
let knnClassifier;
let savedPoses = [];
let isClassifierReady = false;
let currentLandmarks = null;

// Initialisatie
function init() {
    setupUI();
    startPoseDetection();
    initializeML5();
}

// ML5 initialiseren
async function initializeML5() {
    try {
        // Nieuwe methode voor KNNClassifier
        knnClassifier = ml5.KNNClassifier();
        console.log('ML5 KNN Classifier geladen', knnClassifier);
        isClassifierReady = true;
    } catch (error) {
        console.error('Fout bij laden ML5:', error);
        alert('Kon AI model niet initialiseren. Herlaad de pagina.');
    }
}

// UI Opzetten
function setupUI() {
    // Verwijder bestaande knoppen
    document.querySelectorAll('.control-btn').forEach(btn => btn.remove());

    const controls = document.createElement('div');
    controls.className = 'controls';
    
    const buttons = [
        { id: 'exportBtn', text: 'Exporteer naar JSON', onClick: exportToJSON },
        { id: 'trainBtn', text: 'Train Model', onClick: trainKNN },
        { id: 'accuracyBtn', text: 'Bereken Accuracy', onClick: calculateAccuracy }
    ];

    buttons.forEach(btn => {
        const button = document.createElement('button');
        button.className = 'control-btn';
        button.textContent = btn.text;
        button.onclick = btn.onClick;
        controls.appendChild(button);
    });

    const sensitivityControl = document.createElement('div');
    sensitivityControl.className = 'sensitivity-control';
    sensitivityControl.innerHTML = `
        <label for="sensitivity">Gevoeligheid: </label>
        <input type="range" id="sensitivity" min="20" max="50" value="${config.sensitivity}">
        <span id="sensitivityValue">${config.sensitivity}°</span>
    `;
    controls.appendChild(sensitivityControl);

    document.body.insertBefore(controls, videoElement.nextSibling);

    document.getElementById('sensitivity').addEventListener('input', (e) => {
        config.sensitivity = parseInt(e.target.value);
        document.getElementById('sensitivityValue').textContent = `${config.sensitivity}°`;
    });
}

// Pose Detectie
async function startPoseDetection() {
    const pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
    });

    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    pose.onResults(onResults);

    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await pose.send({ image: videoElement });
        },
        width: 640,
        height: 480
    });

    camera.start();
}

// Verwerk pose resultaten
function onResults(results) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    if (results.poseLandmarks) {
        currentLandmarks = results.poseLandmarks;
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2});
        analyzePose(results.poseLandmarks);
    }
}

// Analyseer pose
async function analyzePose(landmarks) {
    const angle = calculatePostureAngle(landmarks);
    const simplifiedData = simplifyPoseData(landmarks).flat();

    if (!isClassifierReady) {
        console.warn('Classifier nog niet klaar');
        basicPostureCheck(angle, landmarks);
        return;
    }

    try {
        const numLabels = await knnClassifier.getNumLabels();
        if (numLabels > 0) {
            const prediction = await knnClassifier.classify(simplifiedData);
            setFeedback(`Voorspelling: ${prediction.label} (${angle.toFixed(1)}°)`, 
                       prediction.label === 'goede_houding' ? 'green' : 'red');
        } else {
            basicPostureCheck(angle, landmarks);
        }
    } catch (error) {
        console.error('Classificatiefout:', error);
        basicPostureCheck(angle, landmarks);
    }
}

function basicPostureCheck(angle, landmarks) {
    if (angle < config.sensitivity) {
        setFeedback(`✓ Goede houding! (${angle.toFixed(1)}°)`, 'green');
        savePoseData(landmarks, "goede_houding");
    } else {
        setFeedback(`⚠️ Slechte houding! (${angle.toFixed(1)}°)`, 'red');
        savePoseData(landmarks, "slechte_houding");
    }
}

function calculatePostureAngle(landmarks) {
    const shoulderAvgY = (landmarks[11].y + landmarks[12].y) / 2;
    const hipAvgY = (landmarks[23].y + landmarks[24].y) / 2;
    return Math.abs(Math.atan2(hipAvgY - shoulderAvgY, 1) * (180 / Math.PI));
}

// Train KNN-model
async function trainKNN() {
    if (!isClassifierReady) {
        alert('AI model is nog niet geladen. Wacht even...');
        return;
    }
    
    if (savedPoses.length < config.minSamples) {
        alert(`Minimaal ${config.minSamples} voorbeelden nodig! Heb nu: ${savedPoses.length}`);
        return;
    }

    try {
        // Reset classifier
        knnClassifier = new ml5.KNNClassifier();
        isClassifierReady = true;
        
        // Voeg voorbeelden toe
        for (const data of savedPoses) {
            await knnClassifier.addExample(data.pose.flat(), data.label);
        }

        setFeedback(`Model getraind met ${savedPoses.length} voorbeelden!`, 'blue');
    } catch (error) {
        console.error('Trainingsfout:', error);
        alert('Trainen mislukt: ' + error.message);
    }
}

// Bereken accuracy
async function calculateAccuracy() {
    if (savedPoses.length < config.minSamples) {
        alert(`Niet genoeg data! Minimaal ${config.minSamples} voorbeelden nodig.`);
        return;
    }

    const testSize = Math.floor(savedPoses.length * 0.2);
    const testData = savedPoses.slice(-testSize);
    
    let correct = 0;
    for (const data of testData) {
        try {
            const prediction = await knnClassifier.classify(data.pose.flat());
            if (prediction.label === data.label) correct++;
        } catch (error) {
            console.error('Classificatiefout:', error);
        }
    }

    const accuracy = (correct / testData.length) * 100;
    setFeedback(`Accuracy: ${accuracy.toFixed(1)}% (${correct}/${testData.length})`, 'blue');
}

// Overige functies
function simplifyPoseData(landmarks) {
    return landmarks.map(landmark => [landmark.x, landmark.y, landmark.z]);
}

function setFeedback(message, color) {
    feedbackElement.textContent = message;
    feedbackElement.style.color = color;
}

function savePoseData(landmarks, label) {
    savedPoses.push({
        label: label,
        pose: simplifyPoseData(landmarks),
        timestamp: new Date().toISOString()
    });
    console.log('Opgeslagen poses:', savedPoses.length);
}

function exportToJSON() {
    if (savedPoses.length === 0) {
        alert("Geen data om te exporteren!");
        return;
    }

    const jsonStr = JSON.stringify(savedPoses, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `houding-data_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

document.addEventListener('DOMContentLoaded', init);