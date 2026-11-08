(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const fileInput = $('fileInput');
  const dropZone = $('dropZone');
  const placeholder = $('dropPlaceholder');
  const preview = $('filePreview');
  const analyzeBtn = $('analyzeBtn');
  const alertBox = $('alertBox');
  const maxBytes = 10 * 1024 * 1024;
  const allowedTypes = ['image/png', 'image/jpeg', 'image/webp'];
  const labelOrder = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe'];

  let chosenFile = null;
  let previewURL = null;
  let isBusy = false;

  const showError = (message) => {
    $('alertText').textContent = message;
    alertBox.hidden = false;
  };
  const hideError = () => { alertBox.hidden = true; };

  const checkStatus = async () => {
    const status = $('modelStatus');
    const text = $('modelStatusText');
    try {
      const response = await fetch('/api/status', {cache: 'no-store'});
      if (!response.ok) throw new Error('Server tidak merespons.');
      const data = await response.json();
      status.classList.remove('is-checking', 'is-ready', 'is-missing');
      status.classList.add(data.model_available ? 'is-ready' : 'is-missing');
      text.textContent = data.model_available ? 'Model terdeteksi' : 'Bobot model belum ada';
    } catch {
      status.classList.remove('is-ready', 'is-checking');
      status.classList.add('is-missing');
      text.textContent = 'Server tidak tersedia';
    }
  };

  const setBusy = (busy) => {
    isBusy = busy;
    analyzeBtn.disabled = busy || !chosenFile;
    analyzeBtn.classList.toggle('is-loading', busy);
    $('analyzeBtnText').textContent = busy ? 'Sedang Menganalisis...' : 'Analisis Gambar';
    $('resultLoading').hidden = !busy;
    if (busy) {
      $('resultEmpty').hidden = true;
      $('resultContent').hidden = true;
      $('visualization').hidden = true;
    }
  };

  const clearResults = () => {
    $('resultEmpty').hidden = false;
    $('resultContent').hidden = true;
    $('resultLoading').hidden = true;
    $('visualization').hidden = true;
  };

  const chooseFile = (file) => {
    if (isBusy) return;
    hideError();
    if (!file) return;
    const extension = (file.name.split('.').pop() || '').toLowerCase();
    if (!allowedTypes.includes(file.type) || !['jpg', 'jpeg', 'png', 'webp'].includes(extension)) {
      showError('Pilih berkas JPG, JPEG, PNG, atau WebP yang valid.');
      return;
    }
    if (file.size > maxBytes) {
      showError('Ukuran gambar maksimum adalah 10 MB.');
      return;
    }
    if (file.size === 0) {
      showError('File gambar kosong.');
      return;
    }
    if (previewURL) URL.revokeObjectURL(previewURL);
    chosenFile = file;
    previewURL = URL.createObjectURL(file);
    $('uploadedImage').src = previewURL;
    $('uploadedName').textContent = file.name;
    $('uploadedSize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Siap dianalisis`;
    placeholder.hidden = true;
    preview.hidden = false;
    analyzeBtn.disabled = false;
    clearResults();
  };

  const reset = () => {
    if (isBusy) return;
    chosenFile = null;
    fileInput.value = '';
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null;
    $('uploadedImage').removeAttribute('src');
    placeholder.hidden = false;
    preview.hidden = true;
    analyzeBtn.disabled = true;
    hideError();
    clearResults();
    $('analisis').scrollIntoView({behavior: 'smooth', block: 'start'});
  };

  const renderResults = (data) => {
    const predicted = data.prediction;
    $('resultGrade').textContent = `Grade ${predicted.grade} · ${predicted.name}`;
    $('resultSeverity').textContent = predicted.severity;
    $('gradeDescription').textContent = predicted.description;
    $('confidenceValue').textContent = `${Number(data.confidence).toFixed(1)}%`;
    const bars = $('probabilityBars');
    bars.replaceChildren();

    data.probabilities.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'prob-row' + (item.grade === predicted.grade ? ' is-top' : '');
      const name = document.createElement('span');
      name.className = 'prob-name';
      name.textContent = `${item.grade} · ${labelOrder[item.grade] || item.name}`;
      const track = document.createElement('div');
      track.className = 'prob-track';
      const fill = document.createElement('div');
      fill.className = 'prob-fill';
      const probability = Math.max(0, Math.min(100, Number(item.probability)));
      fill.style.width = `${probability}%`;
      track.append(fill);
      const value = document.createElement('span');
      value.className = 'prob-value';
      value.textContent = `${probability.toFixed(1)}%`;
      row.append(name, track, value);
      bars.append(row);
    });

    $('resultModelNote').textContent = `Model: ${data.model_name} · Input: ${data.image_size[0]} × ${data.image_size[1]} px · Confidence adalah keluaran model, bukan kepastian diagnosis.`;
    $('resultContent').hidden = false;
    $('resultEmpty').hidden = true;
    $('visualization').hidden = false;
    $('originalResultImg').src = data.original_image;

    const camItems = document.querySelectorAll('.gradcam-only');
    const camOK = Boolean(data.gradcam_available);
    camItems.forEach(item => { item.hidden = !camOK; });
    $('vizImages').style.gridTemplateColumns = camOK ? '' : 'minmax(0, 1fr)';
    $('heatmapResultImg').src = camOK ? data.heatmap_image : '';
    $('overlayResultImg').src = camOK ? data.overlay_image : '';
    $('gradcamNotice').textContent = data.gradcam_error || '';
    $('gradcamNotice').hidden = camOK;
    $('resultsPanel').scrollIntoView({behavior: 'smooth', block: 'center'});
  };

  const runAnalysis = async () => {
    if (!chosenFile || isBusy) return;
    hideError();
    setBusy(true);
    const formData = new FormData();
    formData.append('image', chosenFile, chosenFile.name);
    try {
      const response = await fetch('/api/predict', {method: 'POST', body: formData});
      let data;
      try { data = await response.json(); }
      catch { throw new Error('Respons server tidak valid. Periksa log terminal.'); }
      if (!response.ok) throw new Error(data.error || 'Tidak dapat memproses gambar.');
      renderResults(data);
    } catch (error) {
      clearResults();
      showError(error.message || 'Terjadi gangguan pada analisis gambar.');
      $('alertBox').scrollIntoView({behavior: 'smooth', block: 'center'});
    } finally {
      setBusy(false);
      checkStatus();
    }
  };

  dropZone.addEventListener('click', () => { if (!isBusy) fileInput.click(); });
  dropZone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!isBusy) fileInput.click();
    }
  });
  fileInput.addEventListener('change', event => chooseFile(event.target.files[0]));
  ['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault();
    dropZone.classList.add('drag-active');
  }));
  ['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault();
    dropZone.classList.remove('drag-active');
  }));
  dropZone.addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
  $('dismissAlert').addEventListener('click', hideError);
  analyzeBtn.addEventListener('click', runAnalysis);
  $('resetBtn').addEventListener('click', reset);
  $('printBtn').addEventListener('click', () => window.print());
  checkStatus();
})();
