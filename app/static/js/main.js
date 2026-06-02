(function () {
    'use strict';

    function scheduleAutoReload() {
        var reloadTarget = document.querySelector('[data-auto-reload]');

        if (!reloadTarget) {
            return;
        }

        var delay = Number(reloadTarget.dataset.autoReload);

        if (!Number.isFinite(delay) || delay <= 0) {
            return;
        }

        window.setTimeout(function () {
            window.location.reload();
        }, delay);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scheduleAutoReload);
    } else {
        scheduleAutoReload();
    }
}());
