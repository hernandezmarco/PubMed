// Copyright (C) 2025 Marco Hernandez <ragettyandy@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// For information contact Marco Hernandez <ragettyandy@gmail.com>

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