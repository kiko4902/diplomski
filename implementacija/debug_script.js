
const INFO = {"SMOTE": "Osnovni algoritam — linearna interpolacija između manjinskih primjera.", "Borderline-SMOTE (BS1)": "Fokusira se na granične primjere. BS1 koristi samo manjinske susjede.", "Borderline-SMOTE (BS2)": "Kao BS1, ali dopušta i većinske susjede uz ograničenje.", "ADASYN": "Adaptivno generira više uzoraka u gušćim područjima većinske klase.", "SafeLevel-SMOTE": "Generira samo u 'sigurnim' područjima s dovoljno manjinskih susjeda.", "KMeans-SMOTE": "Grupira manjinske primjere klasterima, više uzoraka u rijetkim klasterima.", "SVM-SMOTE": "Generira uzorke samo oko potpornih vektora na granici odluke.", "SMOTE-ENN": "SMOTE + čišćenje Edited Nearest Neighbors — uklanja šum.", "SMOTE-Tomek": "SMOTE + uklanjanje Tomek Link parova na granici klasa.", "G-SMOTE": "Geometrijsko proširenje — generira unutar sektora, ne samo na liniji.", "Random-SMOTE": "Nasumični smjer i udaljenost za maksimalnu raznolikost.", "PolynomFit-SMOTE": "Polinomna interpolacija kroz više susjeda za nelinearne distribucije."};
let currentTab = 'single';

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
    document.getElementById('sectionSingle').style.display = tab === 'single' ? '' : 'none';
    document.getElementById('sectionCompare').style.display = tab === 'compare' ? '' : 'none';
    document.getElementById('sectionAll').style.display = tab === 'all' ? '' : 'none';
}

function updateInfo() {
    const name = document.getElementById("smote_name").value;
    document.getElementById("info").textContent = INFO[name] || '';
}
updateInfo();

function updateCompareInfo() {
    document.getElementById("info_c1").textContent = INFO[document.getElementById("smote_name_c1").value] || '';
    document.getElementById("info_c2").textContent = INFO[document.getElementById("smote_name_c2").value] || '';
}
updateCompareInfo();

function onDatasetChange() {
    const ds = document.getElementById("dataset_type");
    const opt = ds.options[ds.selectedIndex];
    const isReal = opt.getAttribute("data-real") === "1";
    document.getElementById("n_samples").disabled = isReal;
    document.getElementById("ir").disabled = isReal;
    document.getElementById("noise").disabled = isReal;
    document.getElementById("n_label").textContent = isReal ? "(stvarno)" : "(sinteticki)";
}
onDatasetChange();

async function generate() {
    const btn = document.getElementById("generateBtn");
    btn.disabled = true;
    btn.textContent = "Racunam...";
    document.getElementById("emptyState").style.display = "none";

    const status = document.getElementById("statusArea");
    status.innerHTML = '<div class="status loading">Generiram sinteticke primjere...</div>';

    if (currentTab === 'all') {
        // Batch mode — svi algoritmi, isti dataset
        const params = new URLSearchParams({
            k: document.getElementById("k").value,
            n_samples: document.getElementById("n_samples").value,
            ir: document.getElementById("ir").value,
            noise: document.getElementById("noise").value,
            dim_method: document.getElementById("dim_method").value,
            dataset_type: document.getElementById("dataset_type").value,
            seed: document.getElementById("seed").value,
        });
        try {
            const resp = await fetch("/api/smote_batch?" + params);
            const data = await resp.json();
            if (data.error) { status.innerHTML = '<div class="status error">Greska: ' + data.error + '</div>'; btn.disabled = false; btn.textContent = "Generiraj"; return; }

            status.innerHTML = '<div class="status success">✓ Svi algoritmi na identicnom skupu &nbsp;|&nbsp; ' + data.n_orig + ' originalnih &nbsp;|&nbsp; IR ≈ ' + data.ir.toFixed(1) + ':1 &nbsp;|&nbsp; k=' + data.k + ' &nbsp;|&nbsp; ' + data.dim + '</div>';

            let plotsHtml = '<div class="plots" style="grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));">';
            data.results.forEach((r, i) => {
                const divId = 'plot' + i;
                plotsHtml += '<div class="plot-card" style="padding:8px"><div class="plot-inner" style="height:260px" id="' + divId + '"></div></div>';
            });
            plotsHtml += '</div>';
            document.getElementById("plots").innerHTML = plotsHtml;

            data.results.forEach((r, i) => {
                Plotly.newPlot("plot" + i, JSON.parse(r.plot_json), {}, {responsive: true, displayModeBar: false});
            });
        } catch(e) {
            status.innerHTML = '<div class="status error">Greska: ' + e + '</div>';
        }
    } else {
        // Single or compare mode
        const params = new URLSearchParams({
            smote_name: currentTab === 'compare' ? document.getElementById("smote_name_c1").value : document.getElementById("smote_name").value,
            k: document.getElementById("k").value,
            n_samples: document.getElementById("n_samples").value,
            ir: document.getElementById("ir").value,
            noise: document.getElementById("noise").value,
            dim_method: document.getElementById("dim_method").value,
            smote_name2: currentTab === 'compare' ? document.getElementById("smote_name_c2").value : "",
            dataset_type: document.getElementById("dataset_type").value,
            seed: document.getElementById("seed").value,
        });
        try {
            const resp = await fetch("/api/smote?" + params);
            const data = await resp.json();
            if (data.error) { status.innerHTML = '<div class="status error">Greska: ' + data.error + '</div>'; btn.disabled = false; btn.textContent = "Generiraj"; return; }

            const results = data.results;
            const vname = results[0].name;
            document.getElementById("variantBadge").style.display = "inline";
            document.getElementById("variantBadge").textContent = vname;

            let statusText = '';
            if (results.length === 1) {
                statusText = '<div class="status success">✓ <strong>' + results[0].n_synth + '</strong> synth &nbsp;|&nbsp; ' + data.n_orig + ' orig &nbsp;|&nbsp; IR ≈ ' + data.ir.toFixed(1) + ':1 &nbsp;|&nbsp; ' + data.dim + ' &nbsp;|&nbsp; ' + data.dataset + '</div>';
            } else {
                const d = Math.abs(results[0].n_synth - results[1].n_synth);
                statusText = '<div class="status success">✓ <strong>' + results[0].name + '</strong> (' + results[0].n_synth + ' synth) vs <strong>' + results[1].name + '</strong> (' + results[1].n_synth + ' synth) &nbsp;|&nbsp; Δ=' + d + ' &nbsp;|&nbsp; ' + data.n_orig + ' orig &nbsp;|&nbsp; IR ≈ ' + data.ir.toFixed(1) + ':1</div>';
            }
            status.innerHTML = statusText;

            let plotsHtml = '';
            results.forEach((r, i) => {
                const divId = 'plot' + i;
                plotsHtml += '<div class="plot-card"><div class="plot-meta">' + r.name + ' &middot; ' + r.n_synth + ' synth od ' + r.total + ' ukupno</div><div class="plot-inner" id="' + divId + '"></div></div>';
            });
            document.getElementById("plots").innerHTML = plotsHtml;

            results.forEach((r, i) => {
                Plotly.newPlot("plot" + i, JSON.parse(r.plot_json), {}, {responsive: true, displayModeBar: false});
            });
        } catch(e) {
            status.innerHTML = '<div class="status error">Greska: ' + e + '</div>';
        }
    }

    btn.disabled = false;
    btn.textContent = "Generiraj";
}

        const results = data.results;
        const vname = results[0].name;
        document.getElementById("variantBadge").style.display = "inline";
        document.getElementById("variantBadge").textContent = vname;

        let statusText = '';
        if (results.length === 1) {
            statusText = '<div class="status success">✓ <strong>' + results[0].n_synth + '</strong> sintetičkih primjera &nbsp;|&nbsp; ' + data.n_orig + ' originalnih &nbsp;|&nbsp; IR ≈ ' + data.ir.toFixed(1) + ':1 &nbsp;|&nbsp; ' + data.dim + ' &nbsp;|&nbsp; Dataset: ' + data.dataset + '</div>';
        } else {
            const d = Math.abs(results[0].n_synth - results[1].n_synth);
            statusText = '<div class="status success">✓ <strong>' + results[0].name + '</strong> (' + results[0].n_synth + ' synth) vs <strong>' + results[1].name + '</strong> (' + results[1].n_synth + ' synth) &nbsp;|&nbsp; Δ=' + d + ' &nbsp;|&nbsp; ' + data.n_orig + ' orig &nbsp;|&nbsp; IR ≈ ' + data.ir.toFixed(1) + ':1</div>';
        }
        status.innerHTML = statusText;

        let plotsHtml = '';
        results.forEach((r, i) => {
            const divId = 'plot' + i;
            plotsHtml += '<div class="plot-card"><div class="plot-meta">' + r.name + ' &middot; ' + r.n_synth + ' syntetičkih primjera od ukupno ' + r.total + '</div><div class="plot-inner" id="' + divId + '"></div></div>';
        });
        document.getElementById("plots").innerHTML = plotsHtml;

        results.forEach((r, i) => {
            Plotly.newPlot("plot" + i, JSON.parse(r.plot_json), {}, {responsive: true, displayModeBar: false});
        });

    } catch(e) {
        status.innerHTML = '<div class="status error">Greška u povezivanju: ' + e + '</div>';
    }

    btn.disabled = false;
    btn.textContent = "Generiraj";
}

