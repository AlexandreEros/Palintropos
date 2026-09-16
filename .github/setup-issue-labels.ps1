<#
.SYNOPSIS
    Creates or updates the Palintropos GitHub Issues label taxonomy.

.DESCRIPTION
    This script is the source of truth for the label set described in
    .github/ISSUE_LABELS.md. It is idempotent: every label is applied with
    `gh label create --force`, which creates the label when missing and updates
    its color/description when it already exists. Existing labels outside this
    taxonomy (bug, enhancement, question, ...) are left untouched; nothing is
    ever deleted.

    The one exception is GitHub's default `documentation` label, which is
    *renamed* (not recreated) to `area:documentation` so the namespace stays
    consistent. The rename is skipped if it has already happened.

.PARAMETER Repo
    owner/name of the target repository.

.PARAMETER DryRun
    Print the gh commands instead of running them.

.EXAMPLE
    pwsh .github/setup-issue-labels.ps1 -DryRun
    pwsh .github/setup-issue-labels.ps1
#>
[CmdletBinding()]
param(
    [string]$Repo = 'AlexandreEros/Palintropos',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) not found on PATH."
}

# --- Safety: confirm authentication and that the repo really is the target. ---
$null = gh auth status 2>$null
if ($LASTEXITCODE -ne 0) { throw "gh is not authenticated for github.com. Run: gh auth login" }

$resolved = gh repo view $Repo --json nameWithOwner --jq '.nameWithOwner'
if ($LASTEXITCODE -ne 0 -or $resolved -ne $Repo) {
    throw "Refusing to continue: '$Repo' resolved to '$resolved'."
}
Write-Host "Applying label taxonomy to $resolved" -ForegroundColor Cyan

# --- The taxonomy. Order is cosmetic; namespaces are what matter. ---
$labels = @(
    # Physical engine: which equation set / dynamical core the work belongs to.
    @{ name = 'engine:barotropic-vorticity'; color = '0E4C92'; desc = 'Non-divergent barotropic vorticity core (run/bve, physics/barotropic.py)' },
    @{ name = 'engine:shallow-water';        color = '1F77B4'; desc = 'Rotating shallow-water core (run/swe, physics/shallow_water.py)' },
    @{ name = 'engine:primitive-equations';  color = '5DA2DC'; desc = 'Dry hydrostatic primitive-equation core in sigma coordinates (run/pe)' },

    # Grid backend: the repo's sharpest numerical dividing line.
    @{ name = 'backend:geodesic';     color = '6B4FBB'; desc = 'Icosahedral geodesic grid backend: approximate, orientation-dependent Voronoi quadrature' },
    @{ name = 'backend:gauss-latlon'; color = 'A18AE0'; desc = 'Gauss-Legendre latitude-longitude backend: exact quadrature reference path' },

    # Numerical subsystems that generate work independently of any one engine.
    @{ name = 'numerics:spectral-transforms'; color = '0E8A6E'; desc = 'SH basis, analysis/synthesis, quadrature weights, resolution envelope, spectral operators' },
    @{ name = 'numerics:nonlinear-products';  color = '17B890'; desc = 'Product space, product-grid sampling, dealiasing/truncation, Jacobian and advection products' },
    @{ name = 'numerics:time-integration';    color = '7FD4BE'; desc = 'RK4 and other steppers, CFL ceilings, scheduling, stability margins' },

    # Cross-cutting concerns.
    @{ name = 'area:validation';      color = 'B08900'; desc = 'Benchmarks, analytic test cases, conservation diagnostics, VALIDATION.md evidence' },
    @{ name = 'area:performance';     color = 'C96A00'; desc = 'Runtime cost, throughput, scaling, memory footprint of the numerical path' },
    @{ name = 'area:reproducibility'; color = '8B5E3C'; desc = 'Run capsules, manifests/provenance, determinism, seeds, archived artifacts' },
    @{ name = 'area:visualization';   color = 'D95F9A'; desc = 'Field/spectral plots, comparison figures, normalization, legibility, visualization semantics' },
    @{ name = 'area:cuda';            color = '76B900'; desc = 'GPU-specific behavior: CUDA kernels, CuPy/CUDA versions, driver/TDR limits, device memory' },
    @{ name = 'area:cli';             color = 'C2185B'; desc = 'tropoi command surface, run-config resolution, presets, snapshot scheduling, run lifecycle' },
    @{ name = 'area:documentation';   color = '6E7781'; desc = 'READMEs, docs/, docstrings, and the status prose that describes the model' },

    # How the thing fails. Optional, and not mutually exclusive.
    @{ name = 'failure-mode:silent-wrong-result'; color = '8B0000'; desc = 'Runs to completion and produces plausible output that is numerically or physically wrong' },
    @{ name = 'failure-mode:unsupported-claim';   color = 'A8336A'; desc = 'Implementation may be sound, but the scientific claim made about it lacks evidence' },
    @{ name = 'failure-mode:instability';         color = 'E34F1B'; desc = 'Numerical blow-up, NaN propagation, or an uncontrolled stability margin' },
    @{ name = 'failure-mode:crash';               color = 'F2A9A0'; desc = 'Loud, immediate failure: exception, assertion, OOM, or killed kernel' }
)

# --- Rename GitHub's default `documentation` label into the area: namespace. ---
$existing = gh label list --repo $Repo --limit 200 --json name --jq '.[].name'
if (($existing -contains 'documentation') -and -not ($existing -contains 'area:documentation')) {
    Write-Host "  rename  documentation -> area:documentation"
    if (-not $DryRun) {
        gh label edit 'documentation' --repo $Repo --name 'area:documentation'
        if ($LASTEXITCODE -ne 0) { throw "Failed to rename the 'documentation' label." }
    }
}

# --- Apply every label. `--force` makes this safely rerunnable. ---
foreach ($label in $labels) {
    Write-Host ("  apply   {0}" -f $label.name)
    if ($DryRun) {
        Write-Host ("            gh label create '{0}' --repo {1} --color {2} --description '{3}' --force" -f `
            $label.name, $Repo, $label.color, $label.desc) -ForegroundColor DarkGray
        continue
    }
    gh label create $label.name --repo $Repo --color $label.color --description $label.desc --force
    if ($LASTEXITCODE -ne 0) { throw ("Failed to apply label '{0}'." -f $label.name) }
}

Write-Host "Done. $($labels.Count) labels applied." -ForegroundColor Green
