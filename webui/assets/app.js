const boot = () => {
  import("./app/bootstrap.js")
    .then((mod) => {
      if (mod && typeof mod.bootstrapStudio === "function") {
        return mod.bootstrapStudio();
      }
    })
    .catch((err) => {
      console.error("VisionStack Studio bootstrap failure:", err);
    });
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}