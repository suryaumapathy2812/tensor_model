const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');
const esprima = require('esprima');

const rawData = fs.readFileSync('./src/dataset.json');
const dataset = JSON.parse(rawData);

function tokenizeCode(code) {
    try {
        const tokens = esprima.tokenize(code);
        return tokens.map((token) => token.value);
    } catch (error) {
        console.error(error);
        return [];
    }
}


// const tokenizer = new use.Tokenizer(); // Implement a custom tokenizer for JavaScript code
const tokenizedSamples = dataset.map(sample => tokenizeCode(sample.code));

const topicMapping = {
    "variables": 0,
    "data types": 1,
    "conditionals": 2,
    "loops": 3,
    "functions": 4,
    "arrays": 5,
    "objects": 6,
    "classes": 7,
    "DOM manipulation": 8,
    "AJAX": 9,
    "event handling": 10,
    "error handling": 11,
    "callbacks": 12,
    "promises": 13,
    "async/await": 14,
}

const typeMapping = {
    "VariableDeclaration": 0,
    "Literal": 1,
    "Identifier": 2,
    "IfStatement": 3,
    "SwitchStatement": 4,
    "ConditionalExpression": 5,
    "ForStatement": 6,
    "WhileStatement": 7,
    "DoWhileStatement": 8,
    "ForInStatement": 9,
    "ForOfStatement": 10,
    "FunctionDeclaration": 11,
    "FunctionExpression": 12,
    "ArrowFunctionExpression": 13,
    "ArrayExpression": 14,
    "ObjectExpression": 15,
    "ClassDeclaration": 16,
    "ClassExpression": 17,
    "CallExpression": 18,
    "TryStatement": 19,
    "NewExpression": 20,
}


const oneHotLabels = dataset.map(sample => {
    const oneHotTopicLabel = new Array(Object.keys(topicMapping).length).fill(0);
    sample.labels.topic.forEach(topic => {
        oneHotTopicLabel[topicMapping[topic]] = 1;
    });

    const oneHotTypeLabel = new Array(Object.keys(typeMapping).length).fill(0);
    sample.labels.type.forEach(type => {
        oneHotTypeLabel[typeMapping[type]] = 1;
    });

    return oneHotTopicLabel.concat(oneHotTypeLabel);
});


async function encodeSamples(tokenizedSamples) {
    const model = await use.load();
    const embeddings = await model.embed(tokenizedSamples.map(tokens => tokens.join(' ')));
    return embeddings;
}


function replaceNaNWithZero(value) {
    return isNaN(value) ? 0 : value;
}

function truePositives(yTrue, yPred, classIndex) {
    const condition = yTrue.equal(yPred).cast('float32').mul(yTrue.equal(classIndex).cast('float32'));
    return condition.sum().arraySync();
}

function falsePositives(yTrue, yPred, classIndex) {
    const condition = yTrue.notEqual(yPred).cast('float32').mul(yPred.equal(classIndex).cast('float32'));
    return condition.sum().arraySync();
}

function falseNegatives(yTrue, yPred, classIndex) {
    const condition = yTrue.notEqual(yPred).cast('float32').mul(yTrue.equal(classIndex).cast('float32'));
    return condition.sum().arraySync();
}


function precision(yTrue, yPred, classIndex) {
    const tp = truePositives(yTrue, yPred, classIndex);
    const fp = falsePositives(yTrue, yPred, classIndex);
    return tp / (tp + fp);
}

function recall(yTrue, yPred, classIndex) {
    const tp = truePositives(yTrue, yPred, classIndex);
    const fn = falseNegatives(yTrue, yPred, classIndex);
    return tp / (tp + fn);
}

function f1Score(yTrue, yPred, classIndex) {
    const p = replaceNaNWithZero(precision(yTrue, yPred, classIndex));
    const r = replaceNaNWithZero(recall(yTrue, yPred, classIndex));
    return 2 * ((p * r) / (p + r));
}



(async () => {

    console.log('Encoding samples...');
    const encodedSamples = await encodeSamples(tokenizedSamples);
    console.log('Encoding samples complete.');

    const trainSplit = 0.8;
    const trainSize = Math.floor(encodedSamples.shape[0] * trainSplit);

    const xTrain = encodedSamples.slice(0, trainSize);
    const xVal = encodedSamples.slice(trainSize);

    const y = tf.tensor2d(oneHotLabels);
    const yTrain = y.slice(0, trainSize);
    const yVal = y.slice(trainSize);


    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [encodedSamples.shape[1]] }));
    model.add(tf.layers.dense({ units: Object.keys(topicMapping).length + Object.keys(typeMapping).length, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
        callbacks: [
            tf.node.tensorBoard('logs'),
            {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1} completed. Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`);
                },
            },
        ],
    });

    async function trainModel() {
        const epochs = 10;
        const batchSize = 32;

        await model.fit(xTrain, yTrain, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [xVal, yVal],
            callbacks: tf.node.tensorBoard('logs')
        });

        // Make predictions on the validation set
        const yPred = model.predict(xVal).round();

        // Calculate metrics
        const numClasses = y.shape[1];
        const metrics = [];

        for (let i = 0; i < numClasses; i++) {

            const classAccuracy = yVal
                .equal(yPred)
                .sum(axis = 1)
                .arraySync()
                .filter(value => value === 1)
                .length / yVal.shape[0];

            const classPrecision = replaceNaNWithZero(precision(yVal, yPred, i));
            const classF1 = replaceNaNWithZero(f1Score(yVal, yPred, i));

            metrics.push({
                classIndex: i,
                accuracy: classAccuracy,
                precision: classPrecision,
                f1: classF1,
            });
        }

        console.log(metrics);

    }

    console.log('Training model...');
    await trainModel();
    console.log('Training model complete.');

})()