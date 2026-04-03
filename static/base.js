const bar = document.getElementById('progress-bar');

const Progress = {
  start(durationS = 10, targetPct = 65) {
    bar.classList.remove('error');
    bar.style.display = 'block';
    bar.style.transition = 'none';
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      bar.style.transition = `width ${durationS}s ease`;
      bar.style.width = targetPct + '%';
    });
  },
  set(pct, ms = 400) {
    bar.style.display = 'block';
    bar.style.transition = `width ${ms}ms ease`;
    bar.style.width = pct + '%';
  },
  done() {
    bar.style.transition = 'width .2s ease';
    bar.style.width = '100%';
    setTimeout(() => { bar.style.display = 'none'; bar.style.width = '0%'; }, 350);
  },
  error() {
    bar.classList.add('error');
    this.done();
    setTimeout(() => bar.classList.remove('error'), 600);
  },
};