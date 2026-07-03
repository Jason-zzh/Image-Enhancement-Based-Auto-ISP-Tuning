document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const runBtn = document.getElementById('runBtn');
    const clearBtn = document.getElementById('clearBtn');  // Assume this button exists in HTML
    // const tradImg = document.getElementById('tradImg');
    const origImg = document.getElementById('origImg');
    const enhancedImg = document.getElementById('enhancedImg');
    const status = document.getElementById('status');

    let uploadedFilename = null;

    // Drag/drop/click events (same as before)
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleUpload(files[0]);
    });
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleUpload(e.target.files[0]);
    });

    function handleUpload(file) {
        if (!file.name.toLowerCase().match(/\.(arw|dng|raw)$/)) {
            status.textContent = 'Invalid file type.';
            status.style.color = 'red';
            return;
        }
        uploadedFilename = file.name;
        status.textContent = 'Uploading and processing original...';
        status.style.color = 'orange';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    origImg.src = result.orig_url;
                    origImg.style.display = 'block';
                    // tradImg.src = result.trad_url;
                    // tradImg.style.display = 'block';
                    runBtn.disabled = false;
                    status.textContent = result.message;
                    status.style.color = 'green';
                } else {
                    status.textContent = result.error;
                    status.style.color = 'red';
                    uploadedFilename = null;  // Reset on error
                }
            })
            .catch(error => {
                status.textContent = 'Upload error: ' + error;
                status.style.color = 'red';
                uploadedFilename = null;  // Reset on error
            });
    }

    runBtn.addEventListener('click', async () => {
        if (!uploadedFilename) return;
        
        runBtn.disabled = true;
        status.textContent = 'Running enhancement...';
        status.style.color = 'orange';

        try {
            const response = await fetch('/enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: uploadedFilename })
            });
            const result = await response.json();

            if (result.success) {
                enhancedImg.src = result.enhanced_url;
                enhancedImg.style.display = 'block';
                status.textContent = result.message;
                status.style.color = 'green';
            } else {
                status.textContent = result.error;
                status.style.color = 'red';
            }
        } catch (error) {
            status.textContent = 'Enhance error: ' + error.message;
            status.style.color = 'red';
        } finally {
            runBtn.disabled = false;
        }
    });

    // New: Clear button event listener
    clearBtn.addEventListener('click', () => {
        uploadedFilename = null;
        origImg.src = '';
        origImg.style.display = 'none';
        enhancedImg.src = '';
        enhancedImg.style.display = 'none';
        // tradImg.src = '';
        // tradImg.style.display = 'none';
        runBtn.disabled = true;
        fileInput.value = '';  // Reset file input
        status.textContent = 'Ready to upload a new file.';
        status.style.color = 'black';
        uploadArea.classList.remove('dragover');  // Ensure no dragover state
        console.log('Cleared session.');
    });
});