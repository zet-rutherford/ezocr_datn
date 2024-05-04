const axios = require("axios");
const { Readable } = require("stream");
const FormData = require("form-data");
const path = require("path");
const model = require("../models/index");
const fs = require("fs");

module.exports = {
  index: async (req, res) => {
    console.log(req.user);
    // console.log(req.user.id);
    // const currentUser = await model.User.findOne({ where: { id } });
    // const currentUserName = currentUser.name;
    res.render("image/upload", { req });
  },
  handleUpload: async (req, res) => {
    try {
      const ocrURL = "http://0.0.0.0:8866/extraction/ocr-reader";
      const { image } = req.files;

      let uploadPath =
        path.dirname(path.dirname(__dirname)) + "/public/uploads/" + image.name;
      console.log(uploadPath);
      image.mv(uploadPath);

      console.log(image);
      const stream = Readable.from(image.data);
      stream.path = image.name;
      const formData = new FormData();
      formData.append("image", stream);

      const response = await axios.post(ocrURL, formData, {
        headers: {
          "Content-Type": `multipart/form-data`,
        },
      });
      // console.log("response: ", response.data);
      const result = response.data.data;
      const traceId = response.data.trace_id;
      // console.log(result);
      // res.send({ data: response.data });
      const content = result.join("\n");
      console.log(content);
      const filepath = "/uploads/" + image.name;
      const newImage = await model.Image.create({
        filepath,
        content,
        userId: req.user.id,
      });
      const imageId = newImage.id;
      console.log(newImage);
      if (!result.length) {
        res.render("image/fail");
      }
      res.render("image/result", { imageId, content, filepath, req });
    } catch (error) {
      console.log(error);
    }
  },
  getHistory: async (req, res) => {
    const { id } = req.params;
    const listImage = await model.Image.findAll({ where: { userId: id } });
    console.log(listImage);
    if (!listImage.length) {
      res.render("image/emptyHistory");
    }
    res.render("image/history", { listImage, req });
  },
  getDetail: async (req, res) => {
    const { id } = req.params;
    const currentImage = await model.Image.findOne({ where: { id } });
    console.log(currentImage);
    if (!currentImage) {
      res.render("image/emptyHistory");
    }
    res.render("image/detail", { currentImage, req });
  },
  delete: async (req, res) => {
    const { id } = req.params;
    const selectedImage = await model.Image.findOne({ where: { id } });
    const uploadPath =
      path.dirname(path.dirname(__dirname)) +
      "/public" +
      selectedImage.filepath;
    fs.unlinkSync(uploadPath);
    await model.Image.destroy({ where: { id } });
    res.redirect("/history/" + req.user.id);
  },
};
