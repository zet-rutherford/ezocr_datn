const express = require("express");
const imageController = require("../controllers/image.controller");

const router = express.Router();

/* GET home page. */
router.get("/", imageController.index);
router.post("/upload", imageController.handleUpload);
router.get("/history/:id", imageController.getHistory);
router.get("/detail/:id", imageController.getDetail);
router.get("/delete/:id", imageController.delete);

module.exports = router;
