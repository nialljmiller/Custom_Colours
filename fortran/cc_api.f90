! ***********************************************************************
! cc_api.f90
!
! Thin f2py-bindable wrappers around cc_kernels.
!
! Rules for clean f2py bindings:
!   - No derived types in any public interface
!   - All array dimensions passed explicitly as integers
!   - Intent annotations on every dummy argument
!   - No allocatable intent(out) ? caller allocates, Fortran fills
!   - Module-level USE of cc_kernels; no re-export of internals
!
! Python import after building:
!   import cc_api
!   cc_api.interp_sed(teff, logg, meta, teff_grid, logg_grid, meta_grid,
!                     flux_cube, result_flux, ierr)
! ***********************************************************************

module cc_api
   use cc_kernels, only: dp, &
      cc_hermite_interp_vector, &
      cc_trilinear_interp_vector, &
      cc_dilute_flux, &
      cc_simpson_integration, &
      cc_trapz_integration, &
      cc_interp_filter_onto_sed, &
      cc_synthetic_flux, &
      cc_magnitude, &
      cc_bolometric_flux, &
      cc_bolometric_magnitude, &
      cc_vega_zero_point, &
      cc_ab_zero_point, &
      cc_st_zero_point

   implicit none
   private

   public :: interp_sed_hermite
   public :: interp_sed_linear
   public :: dilute_flux
   public :: synthetic_magnitude
   public :: bolometric
   public :: vega_zero_point
   public :: ab_zero_point
   public :: st_zero_point
   public :: trapz
   public :: simpson

contains

   ! -------------------------------------------------------------------------
   ! interp_sed_hermite
   !
   ! Interpolate a full SED at (teff, logg, meta) using cubic Hermite
   ! tensor interpolation on the preloaded flux cube.
   !
   ! Parameters (Python-visible names after f2py binding)
   ! -------------------------------------------------------
   ! teff, logg, meta : scalar float64 ? query point
   ! teff_grid(nt)    : float64 array ? Teff axis
   ! logg_grid(nl)    : float64 array ? logg axis
   ! meta_grid(nm)    : float64 array ? [M/H] axis
   ! flux_cube(nt,nl,nm,nw) : float64 ? the full atmosphere flux cube
   ! result_flux(nw)  : float64, intent(out) ? interpolated surface flux
   ! ierr             : int, intent(out) ? 0=ok, 1=clamped to boundary
   ! -------------------------------------------------------------------------
   subroutine interp_sed_hermite(teff, logg, meta, &
                                  teff_grid, nt, &
                                  logg_grid, nl, &
                                  meta_grid, nm, &
                                  flux_cube, nw, &
                                  result_flux, ierr)
      !f2py intent(in)  :: teff, logg, meta
      !f2py intent(in)  :: teff_grid, logg_grid, meta_grid
      !f2py intent(in)  :: nt, nl, nm, nw
      !f2py intent(in)  :: flux_cube
      !f2py intent(out) :: result_flux
      !f2py intent(out) :: ierr
      real(dp), intent(in)  :: teff, logg, meta
      integer,  intent(in)  :: nt, nl, nm, nw
      real(dp), intent(in)  :: teff_grid(nt), logg_grid(nl), meta_grid(nm)
      real(dp), intent(in)  :: flux_cube(nt, nl, nm, nw)
      real(dp), intent(out) :: result_flux(nw)
      integer,  intent(out) :: ierr

      call cc_hermite_interp_vector(teff, logg, meta, &
                                    teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                    flux_cube, nw, result_flux, ierr)
   end subroutine interp_sed_hermite


   ! -------------------------------------------------------------------------
   ! interp_sed_linear
   !
   ! Same interface as interp_sed_hermite but using trilinear interpolation.
   ! -------------------------------------------------------------------------
   subroutine interp_sed_linear(teff, logg, meta, &
                                 teff_grid, nt, &
                                 logg_grid, nl, &
                                 meta_grid, nm, &
                                 flux_cube, nw, &
                                 result_flux, ierr)
      !f2py intent(in)  :: teff, logg, meta
      !f2py intent(in)  :: teff_grid, logg_grid, meta_grid
      !f2py intent(in)  :: nt, nl, nm, nw
      !f2py intent(in)  :: flux_cube
      !f2py intent(out) :: result_flux
      !f2py intent(out) :: ierr
      real(dp), intent(in)  :: teff, logg, meta
      integer,  intent(in)  :: nt, nl, nm, nw
      real(dp), intent(in)  :: teff_grid(nt), logg_grid(nl), meta_grid(nm)
      real(dp), intent(in)  :: flux_cube(nt, nl, nm, nw)
      real(dp), intent(out) :: result_flux(nw)
      integer,  intent(out) :: ierr

      call cc_trilinear_interp_vector(teff, logg, meta, &
                                      teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                      flux_cube, nw, result_flux, ierr)
   end subroutine interp_sed_linear


   ! -------------------------------------------------------------------------
   ! dilute_flux
   !
   ! Apply (R/d)^2 dilution to convert surface flux to observed flux.
   ! -------------------------------------------------------------------------
   subroutine dilute_flux(surface_flux, nw, R, d, observed_flux)
      !f2py intent(in)  :: surface_flux, nw, R, d
      !f2py intent(out) :: observed_flux
      integer,  intent(in)  :: nw
      real(dp), intent(in)  :: surface_flux(nw), R, d
      real(dp), intent(out) :: observed_flux(nw)
      call cc_dilute_flux(surface_flux, nw, R, d, observed_flux)
   end subroutine dilute_flux


   ! -------------------------------------------------------------------------
   ! synthetic_magnitude
   !
   ! Compute a synthetic magnitude in a single filter from a diluted SED.
   !
   ! Steps performed internally:
   !   1. Interpolate filter transmission onto the SED wavelength grid
   !   2. Photon-counting in-band flux integration
   !   3. m = -2.5 log10(F_band / F_zp)
   !
   ! Parameters
   ! ----------
   ! sed_wave(nw)        : SED wavelength grid (?)
   ! obs_flux(nw)        : diluted (observer-frame) SED flux (erg/s/cm^2/?)
   ! filt_wave(nf)       : filter wavelength grid (?)
   ! filt_trans(nf)      : filter transmission [0?1]
   ! zero_point          : precomputed photometric zero-point
   ! mag                 : output magnitude
   ! band_flux           : output in-band flux (before zero-point)
   ! ierr                : 0=ok, 1=integration failure, 2=non-positive flux
   ! -------------------------------------------------------------------------
   subroutine synthetic_magnitude(sed_wave, obs_flux, nw, &
                                   filt_wave, filt_trans, nf, &
                                   zero_point, &
                                   mag, band_flux, ierr)
      !f2py intent(in)  :: sed_wave, obs_flux, nw
      !f2py intent(in)  :: filt_wave, filt_trans, nf
      !f2py intent(in)  :: zero_point
      !f2py intent(out) :: mag, band_flux, ierr
      integer,  intent(in)  :: nw, nf
      real(dp), intent(in)  :: sed_wave(nw), obs_flux(nw)
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(in)  :: zero_point
      real(dp), intent(out) :: mag, band_flux
      integer,  intent(out) :: ierr

      real(dp), dimension(nw) :: filt_on_sed
      integer :: ierr2

      call cc_interp_filter_onto_sed(filt_wave, filt_trans, nf, &
                                      sed_wave, nw, filt_on_sed, ierr)
      if (ierr /= 0) then
         mag = -99.9_dp; band_flux = -1.0_dp; return
      end if

      call cc_synthetic_flux(sed_wave, obs_flux, filt_on_sed, nw, band_flux, ierr)
      if (ierr /= 0) then
         mag = -99.9_dp; return
      end if

      call cc_magnitude(band_flux, zero_point, mag, ierr2)
      if (ierr2 /= 0) ierr = ierr2
   end subroutine synthetic_magnitude


   ! -------------------------------------------------------------------------
   ! bolometric
   !
   ! Compute bolometric flux and magnitude from a diluted SED.
   ! -------------------------------------------------------------------------
   subroutine bolometric(sed_wave, obs_flux, nw, bol_flux, bol_mag, ierr)
      !f2py intent(in)  :: sed_wave, obs_flux, nw
      !f2py intent(out) :: bol_flux, bol_mag, ierr
      integer,  intent(in)  :: nw
      real(dp), intent(in)  :: sed_wave(nw), obs_flux(nw)
      real(dp), intent(out) :: bol_flux, bol_mag
      integer,  intent(out) :: ierr

      integer :: ierr2
      call cc_bolometric_flux(sed_wave, obs_flux, nw, bol_flux, ierr)
      call cc_bolometric_magnitude(bol_flux, bol_mag, ierr2)
      if (ierr2 /= 0 .and. ierr == 0) ierr = ierr2
   end subroutine bolometric


   ! -------------------------------------------------------------------------
   ! Zero-point subroutines  (called once at init from Python)
   ! -------------------------------------------------------------------------

   subroutine vega_zero_point(vega_wave, vega_flux, nv, &
                               filt_wave, filt_trans, nf, &
                               zp, ierr)
      !f2py intent(in)  :: vega_wave, vega_flux, nv
      !f2py intent(in)  :: filt_wave, filt_trans, nf
      !f2py intent(out) :: zp, ierr
      integer,  intent(in)  :: nv, nf
      real(dp), intent(in)  :: vega_wave(nv), vega_flux(nv)
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr
      call cc_vega_zero_point(vega_wave, vega_flux, nv, filt_wave, filt_trans, nf, zp, ierr)
   end subroutine vega_zero_point


   subroutine ab_zero_point(filt_wave, filt_trans, nf, zp, ierr)
      !f2py intent(in)  :: filt_wave, filt_trans, nf
      !f2py intent(out) :: zp, ierr
      integer,  intent(in)  :: nf
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr
      call cc_ab_zero_point(filt_wave, filt_trans, nf, zp, ierr)
   end subroutine ab_zero_point


   subroutine st_zero_point(filt_wave, filt_trans, nf, zp, ierr)
      !f2py intent(in)  :: filt_wave, filt_trans, nf
      !f2py intent(out) :: zp, ierr
      integer,  intent(in)  :: nf
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr
      call cc_st_zero_point(filt_wave, filt_trans, nf, zp, ierr)
   end subroutine st_zero_point


   ! -------------------------------------------------------------------------
   ! Standalone integration (exposed for testing / Python use)
   ! -------------------------------------------------------------------------

   subroutine trapz(x, y, n, result)
      !f2py intent(in)  :: x, y, n
      !f2py intent(out) :: result
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: x(n), y(n)
      real(dp), intent(out) :: result
      call cc_trapz_integration(x, y, n, result)
   end subroutine trapz


   subroutine simpson(x, y, n, result)
      !f2py intent(in)  :: x, y, n
      !f2py intent(out) :: result
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: x(n), y(n)
      real(dp), intent(out) :: result
      call cc_simpson_integration(x, y, n, result)
   end subroutine simpson

end module cc_api
