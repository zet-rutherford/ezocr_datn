const model = require("../models/index");
const User = model.User;
const hash = require("../helpers/hash");
const mailer = require("../helpers/mailer");
const jwt = require("jsonwebtoken");

module.exports = {
  login: async (req, res) => {
    const msg = req.flash("error");
    const msgType = msg ? "danger" : "success";
    return res.render("auth/login", {
      pageTitle: "Đăng nhập",
      msg,
      msgType,
      layout: "layouts/auth.ejs",
    });
  },

  handleLogin: async (req, res) => {
    return res.redirect("/");
  },
  logout: (req, res, next) => {
    req.logout(function (err) {
      if (err) {
        return next(err);
      }
      res.redirect("/auth/login");
    });
  },
  register: (req, res) => {
    return res.render("auth/register", { layout: "layouts/auth.ejs" });
  },
  handleRegister: async (req, res) => {
    const { email, name, password } = req.body;
    console.log(email, name, password);
    await model.User.create({
      email,
      name,
      password: hash.make(password),
    });
    return res.redirect("/auth/login");
  },

  forgot: (req, res) => {
    return res.render("auth/forgotPassword", { layout: "layouts/auth.ejs" });
  },
  handleForgot: async (req, res) => {
    const { email } = req.body;
    const user = model.User.findOne({ where: { email } });
    if (user) {
      const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET, {
        expiresIn: "15m",
      });
      mailer.sendForgot(email, token);
      return res.render("auth/resetPassword", {
        layout: "layouts/auth.ejs",
      });
    }
  },
  reset: (req, res) => {
    return res.render("auth/resetPassword", {
      layout: "layouts/auth.ejs",
    });
  },
  handleReset: async (req, res) => {
    const { token } = req.params;
    const { passwordNew, rePassword } = req.body;
    if (!passwordNew || !rePassword) {
      req.flash("error", "Vui lòng nhập đầy đủ thông tin");
      res.redirect("/forgot-password/verify/" + token);
    } else if (passwordNew !== rePassword) {
      req.flash("error", "Mật khẩu nhập lại không khớp");
      res.redirect("/forgot-password/verify/" + token);
    } else {
      var decoded = jwt.verify(token, "secret");
      if (decoded) {
        const salt = 10;
        bcrypt.hash(passwordNew, salt, async function (err, hash) {
          await model.User.update(
            { password: hash },
            {
              where: {
                id: decoded.id,
              },
            }
          );
        });

        res.redirect("/auth/login");
      } else {
        res.send("<h1>Link xác thực đã hết hạn hoặc không tồn tại</h1>");
      }
    }
  },
};
