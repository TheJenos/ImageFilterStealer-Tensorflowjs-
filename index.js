
var smallernum = 10;
var loop = true;

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

function readURL(input, imagetag, size) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var myCanvas = document.getElementById(imagetag);
            var ctx = myCanvas.getContext('2d');
            var img = new Image;
            img.onload = function () {
                myCanvas.width = img.width / size;
                myCanvas.height = img.height / size;
                ctx.drawImage(img, 0, 0, img.width / size, img.height / size);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
}
function trainloop() {
    loop = true;
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

    let imageins_array = [];
    let imageouts_array = [];

    for (let index = 0; index < input_t.width; index++) {
        for (let index1 = 0; index1 < input_t.height; index1++) {

            var x = (index / input_t.width);
            var y = (index1 / input_t.height);

            var pixelData1 = input_t_ctx.getImageData(index, index1, 1, 1).data;
            var r1 = (pixelData1[0] / 255);
            var g1 = (pixelData1[1] / 255);
            var b1 = (pixelData1[2] / 255);
            var a1 = (pixelData1[3] / 255);

            var pixelData2 = out_t_ctx.getImageData(index, index1, 1, 1).data;
            var r2 = (pixelData2[0] / 255);
            var g2 = (pixelData2[1] / 255);
            var b2 = (pixelData2[2] / 255);
            var a2 = (pixelData2[3] / 255);

            imageins_array.push([r1, g1, b1, a1]);
            imageouts_array.push([r2, g2, b2, a2]);

        }
    }

    const ins = tf.tensor2d(imageins_array);
    const outs = tf.tensor2d(imageouts_array);


    var train = async function () {
        const resp = await model.fit(ins, outs, {
            epochs: 10
        });
        rate = resp.history.loss[0];
        document.getElementById('loss').innerText = resp.history.loss[0];
        //console.log(resp.history.loss[0]);
    };

    var backtrain = true;
    var trainit = function(){
        if(backtrain)train().then(() => {
            backtrain = true;
        });
        backtrain = false;
    }

    var batch_count = 200;
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
        if (index > maxcount) {
            if(loop) {
                out_d_ctx.clearRect(0, 0, input_d.width, input_d.height);
                trainit();
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