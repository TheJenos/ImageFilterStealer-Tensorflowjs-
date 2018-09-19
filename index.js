
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
const hiddenLayer2 = tf.layers.dense({
    units: 20,
    activation: 'sigmoid'
});
model.add(hiddenLayer2);
const hiddenLayer3 = tf.layers.dense({
    units: 20,
    activation: 'sigmoid'
});
model.add(hiddenLayer3);
const outputLayer = tf.layers.dense({
    units: 4,
    activation: 'sigmoid'
});
model.add(outputLayer);

const optimizer = tf.train.adam(0.01);

model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError'
});

function stop() {
    document.getElementById('stop-btn').style.display = 'none';
    document.getElementById('sample-btn').style.display = 'inline';
    document.getElementById('run-btn').style.display = 'inline';
    loop = false;
}

function init(index) {
    var cans = ['input_t', 'out_t', 'input_d'];
    var myCanvas = document.getElementById(cans[index]);
    var ctx = myCanvas.getContext('2d');
    var img = new Image;
    img.onload = function () {
        myCanvas.width = img.width / smallernum;
        myCanvas.height = img.height / smallernum;
        ctx.drawImage(img, 0, 0, img.width / smallernum, img.height / smallernum);
        if (index + 1 < cans.length) {
            init(index + 1);
        } else {
            trainloop();
        }
    };
    img.src = "image.jpg";
}

function readURL(input, imagetag) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var myCanvas = document.getElementById(imagetag);
            var ctx = myCanvas.getContext('2d');
            var img = new Image;
            img.onload = function () {
                myCanvas.width = img.width / smallernum;
                myCanvas.height = img.height / smallernum;
                ctx.drawImage(img, 0, 0, img.width / smallernum, img.height / smallernum);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function update() {
    in_batch_count = parseInt(document.getElementById('inbatch').value);
    batch_count = parseInt(document.getElementById('outbatch').value);
    smallernum = 1 / parseFloat(document.getElementById('scaler').value);
}

function trainloop() {
    loop = true;

    document.getElementById('stop-btn').style.display = 'inline';
    document.getElementById('sample-btn').style.display = 'none';
    document.getElementById('run-btn').style.display = 'none';

    var input_t = document.getElementById('input_t');
    var out_t = document.getElementById('out_t');
    var input_t_ctx = input_t.getContext("2d");
    var out_t_ctx = out_t.getContext("2d");

    var input_d = document.getElementById('input_d');
    var out_d = document.getElementById('out_d');
    var input_d_ctx = input_d.getContext("2d");
    var out_d_ctx = out_d.getContext("2d");

    out_d.width = input_d.width
    out_d.height = input_d.height


    var index_t = 1;
    var train = async function () {
        var rowcolors = [];
        var rowcolors_2 = [];
        for (let index1 = 0; index1 < in_batch_count; index1++) {
            var pos = getIJ(input_d.width, input_d.height, index_t + index1);
            var pixelData1 = input_t_ctx.getImageData(pos[0], pos[1], 1, 1).data;
            var r1 = (pixelData1[0] / 255);
            var g1 = (pixelData1[1] / 255);
            var b1 = (pixelData1[2] / 255);
            var a1 = (pixelData1[3] / 255);
            rowcolors.push([r1, g1, b1, a1]);

            var pixelData2 = out_t_ctx.getImageData(pos[0], pos[1], 1, 1).data;
            var r2 = (pixelData2[0] / 255);
            var g2 = (pixelData2[1] / 255);
            var b2 = (pixelData2[2] / 255);
            var a2 = (pixelData2[3] / 255);

            rowcolors_2.push([r1, g1, b1, a1]);
        }

        const ins = tf.tensor2d(rowcolors);
        const outs = tf.tensor2d(rowcolors_2);


        const resp = await model.fit(ins, outs, {
            epochs: 1
        });

        rate = resp.history.loss[0];
        document.getElementById('loss').innerText = resp.history.loss[0];

        ins.dispose();
        outs.dispose();

        index_t += in_batch_count;
        if (index_t > maxcount) {
            index_t = 1;
        }

    };


    var trainit = function () {
        if (loop) train().then(() => {
            trainit();
        });
    }

    trainit();

    var index = 1;
    var maxcount = input_d.width * input_d.height;
    var run = function () {
        var rowcolors = [];
        for (let index1 = 0; index1 < batch_count; index1++) {
            var pos = getIJ(input_d.width, input_d.height, index + index1);
            var pixelData1 = input_d_ctx.getImageData(pos[0], pos[1], 1, 1).data;
            var r1 = (pixelData1[0] / 255);
            var g1 = (pixelData1[1] / 255);
            var b1 = (pixelData1[2] / 255);
            var a1 = (pixelData1[3] / 255);
            rowcolors.push([r1, g1, b1, a1]);
        }
        const ins_d = tf.tensor2d(rowcolors);
        tf.tidy(() => {
            var outputs = model.predict(ins_d).dataSync();
            var split = 0;
            for (let index1 = 0; index1 < batch_count; index1++) {
                var pos = getIJ(input_d.width, input_d.height, index + index1);
                var r = Math.round(outputs[split + 0] * 255);
                var g = Math.round(outputs[split + 1] * 255);
                var b = Math.round(outputs[split + 2] * 255);
                var a = Math.round(outputs[split + 3] * 255);
                out_d_ctx.fillStyle = 'rgba(' + r + ', ' + g + ', ' + b + ', ' + a + ')';
                out_d_ctx.fillRect(pos[0], pos[1], 1, 1);
                split += 4;
            }
        });
        ins_d.dispose();
        index += batch_count;
        document.getElementById('pp').innerText = Math.round((index / maxcount) * 100) + "%";
        if (index > maxcount) {
            if (loop) {
                out_d_ctx.clearRect(0, 0, input_d.width, input_d.height);
            }
            index = 1;
        }
        if (loop || index != 1) setTimeout(run, 10);
    }
    setTimeout(run, 10);

}

function getIJ(width, height, index) {
    var row = Math.trunc(index / width);
    var col = index - (row * width);
    return [col, row];
}