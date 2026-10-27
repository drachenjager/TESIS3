/**
 * main.js — lógica de la interfaz de Pronóstico USD/MXN
 *
 * Responsabilidades:
 *   1. Inicializar DataTables en la tabla de métricas.
 *   2. Alternar entre el modo de consulta "periodo" y "fechas personalizadas".
 *   3. Pre-rellenar las fechas del modo personalizado con valores razonables.
 *   4. Mostrar un spinner mientras el formulario principal está procesando.
 */

$(document).ready(function () {

    // -----------------------------------------------------------------------
    // 1. DataTables
    // -----------------------------------------------------------------------
    if ($('#metrics-table').length) {
        $('#metrics-table').DataTable({
            paging:   false,
            info:     false,
            searching: false,
            ordering: true,
        });
    }

    // -----------------------------------------------------------------------
    // 2. Toggle entre paneles de modo de consulta
    // -----------------------------------------------------------------------
    function applyMode(mode) {
        $('#panel_period').hide();
        $('#panel_anchor').hide();
        $('#panel_custom').hide();
        if (mode === 'custom') {
            $('#panel_custom').show();
        } else if (mode === 'anchor') {
            $('#panel_anchor').show();
        } else {
            $('#panel_period').show();
        }
    }

    $('input[name="query_mode"]').on('change', function () {
        applyMode(this.value);
    });

    // Aplica el estado inicial (por si el servidor devuelve un modo distinto al default)
    var initialMode = $('input[name="query_mode"]:checked').val() || 'period';
    applyMode(initialMode);

    // -----------------------------------------------------------------------
    // 3. Pre-rellenar fechas del modo personalizado
    // -----------------------------------------------------------------------
    var today = new Date();
    var todayStr = today.toISOString().split('T')[0];

    // Fecha máxima = hoy (para todos los campos de fecha)
    $('#start_date, #end_date, #anchor_date').attr('max', todayStr);

    // Pre-rellena anchor_date con hoy si está vacío
    if (!$('#anchor_date').val()) {
        $('#anchor_date').val(todayStr);
    }

    // Solo pre-rellena si el campo está vacío (no sobrescribe un valor devuelto
    // por el servidor tras un error de validación)
    if (!$('#end_date').val()) {
        $('#end_date').val(todayStr);
    }
    if (!$('#start_date').val()) {
        var oneYearAgo = new Date(today);
        oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
        $('#start_date').val(oneYearAgo.toISOString().split('T')[0]);
    }

    // Evita que end_date sea anterior a start_date
    $('#start_date').on('change', function () {
        if ($('#end_date').val() && $('#end_date').val() < this.value) {
            $('#end_date').val(this.value);
        }
        $('#end_date').attr('min', this.value);
    });

    // -----------------------------------------------------------------------
    // 4. Spinner de carga
    // -----------------------------------------------------------------------
    $('#main-form').on('submit', function () {
        $('#btn-submit').prop('disabled', true).text('Procesando…');
        $('#loading-overlay').removeClass('d-none');
    });

    // -----------------------------------------------------------------------
    // 5. Sincronizar modelo seleccionado con formulario de residuales
    // -----------------------------------------------------------------------
    function syncModelChoice() {
        var val = $('input[name="model_choice"]:checked').val() || '';
        $('#residuals-model-input').val(val);
    }
    $('input[name="model_choice"]').on('change', syncModelChoice);
    syncModelChoice();

});
