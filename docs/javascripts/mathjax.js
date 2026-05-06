window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  const mj = window.MathJax;
  if (!mj || !mj.startup || typeof mj.startup.promise === "undefined") {
    return;
  }
  mj.startup.promise
    .then(() => {
      mj.startup.output?.clearCache?.();
      mj.typesetClear?.();
      mj.texReset?.();
      return mj.typesetPromise();
    })
    .catch((err) => {
      console.warn("MathJax typeset:", err);
    });
});
