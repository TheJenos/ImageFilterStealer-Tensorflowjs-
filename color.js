var smallernum = 1;
var loop = true;
var batch_count = 200;
var in_batch_count = 200;

let model = tf.sequential();

const hiddenLayer = tf.layers.dense({
    units: 20,
    inputShape: [4],
    activation: 'sigmoid'
});
model.add(hiddenLayer);
const hiddenLayer1 = tf.layers.dense({
    units: 10,
    activation: 'sigmoid'
});
model.add(hiddenLayer1);
const outputLayer = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
});
model.add(outputLayer);

const optimizer = tf.train.adam(0.01);

model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError'
});

function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

const dataset = {
    gray: {
        50: "#f9fafb",
        100: "#f3f4f6",
        200: "#e5e7eb",
        300: "#d1d5db",
        400: "#9ca3af",
        500: "#6b7280",
        600: "#4b5563",
        700: "#374151",
        800: "#1f2937",
        900: "#111827",
    },
    amber: {
        50: "#fffbeb",
        100: "#fff3c7",
        200: "#fde68a",
        300: "#fcd34d",
        400: "#fbbf24",
        500: "#f59e0b",
        600: "#d97706",
        700: "#b45309",
        800: "#92400e",
        900: "#78350f",
    },
    pink: {
        50: "#fdf2f8",
        100: "#fce7f3",
        200: "#fbcfe8",
        300: "#f9a8d4",
        400: "#f472b6",
        500: "#ec4899",
        600: "#db2777",
        700: "#be185d",
        800: "#9d174d",
        900: "#831843",
    },
    blue: {
        50: "#eff6ff",
        100: "#dbeafe",
        200: "#bfdbfe",
        300: "#93c5fd",
        400: "#60a5fa",
        500: "#3b82f6",
        600: "#2563eb",
        700: "#1d4ed8",
        800: "#1e40af",
        900: "#1e3a8a",
    }
}

var colorIWant = '#6366f1';

function stop() {
    document.getElementById('stop-btn').style.display = 'none';
    document.getElementById('run-btn').style.display = 'inline';
    loop = false;
}

function trainloop() {

    var colorInput = document.getElementById('color');
    colorIWant = colorInput.value;

    loop = true;

    document.getElementById('stop-btn').style.display = 'inline';
    document.getElementById('run-btn').style.display = 'none';

    var train = async function () {
        var baseColorAndShade = [];
        var outColors = [];

        for (const key in dataset) {
            const baseColor = hexToRgb(dataset[key]['500']);
            for (const shade in dataset[key]) {
                const normalizedInputRed = baseColor.r / 255;
                const normalizedInputGreen = baseColor.g / 255;
                const normalizedInputBlue = baseColor.b / 255;
                const normalizedInputShade = parseInt(shade) / 900;
                baseColorAndShade.push([normalizedInputRed, normalizedInputGreen, normalizedInputBlue, normalizedInputShade]);

                const outColor = hexToRgb(dataset[key][shade]);
                const normalizedOutputRed = outColor.r / 255;
                const normalizedOutputGreen = outColor.g / 255;
                const normalizedOutputBlue = outColor.b / 255;
                outColors.push([normalizedOutputRed, normalizedOutputGreen, normalizedOutputBlue]);
            }
        }

        const ins = tf.tensor2d(baseColorAndShade);
        const outs = tf.tensor2d(outColors);

        const resp = await model.fit(ins, outs, {
            epochs: 1
        });

        rate = resp.history.loss[0];
        document.getElementById('loss').innerText = resp.history.loss[0];
        console.log(resp.history.loss[0]);

        ins.dispose();
        outs.dispose();
    };


    var trainInit = function () {
        if (loop) train().then(() => {
            trainInit();
        });
    }

    trainInit();

    const shades = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900'];

    var index = 1;
    var run = function () {

        var baseColorAndShade = [];
        const baseColor = hexToRgb(colorIWant);
        for (const shade of shades) {
            const normalizedInputRed = baseColor.r / 255;
            const normalizedInputGreen = baseColor.g / 255;
            const normalizedInputBlue = baseColor.b / 255;
            const normalizedInputShade = parseInt(shade) / 900;
            baseColorAndShade.push([normalizedInputRed, normalizedInputGreen, normalizedInputBlue, normalizedInputShade]);
        }

        const ins_d = tf.tensor2d(baseColorAndShade);

        tf.tidy(() => {
            var outputs = model.predict(ins_d).dataSync();

            const output = {}
            var offset = 0
            
            for (const shade of shades) {
                const red = Math.round(outputs[offset] * 255);
                const green = Math.round(outputs[offset + 1] * 255);
                const blue = Math.round(outputs[offset + 2] * 255);

                const hex = rgbToHex(red, green, blue);
                output[shade] = hex;

                const cell = document.getElementById(shade+"_COLOR");
                const cellText = document.getElementById(shade);
                cell.style.backgroundColor = hex;
                cellText.innerText = hex;

                offset += 3;
            }

            
        });

        ins_d.dispose();

        index += batch_count;
        if (loop || index != 1) setTimeout(run, 10);
    }

    setTimeout(run, 10)
}
