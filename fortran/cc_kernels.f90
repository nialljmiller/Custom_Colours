! ***********************************************************************
! cc_kernels.f90
!
! MESA-free numerical kernels for Custom_Colours.
! Ported from colors/private/{hermite_interp,linear_interp,
!   colors_utils,synthetic,bolometric}.f90 with all MESA-specific
! types, handle infrastructure, and I/O removed.
!
! All subroutines operate purely on arrays passed by the caller.
! No global state.  No disk I/O.  No STOP on error — ierr flags used.
!
! Copyright (C) 2025 Niall Miller
! LGPL-3.0-or-later
! ***********************************************************************

module cc_kernels
   implicit none
   private

   integer, parameter :: dp = kind(1.0d0)

   ! Bolometric constants (cgs)
   real(dp), parameter :: CLIGHT_CM_S  = 2.99792458d10
   real(dp), parameter :: AB_FNU_ZP    = 3.631d-20      ! 3631 Jy in erg/s/cm^2/Hz
   real(dp), parameter :: ST_FLAM_ZP   = 3.63d-9        ! flat f_lambda zp erg/s/cm^2/A
   real(dp), parameter :: M_BOL_SUN    = 4.74d0         ! solar bolometric absolute mag
   real(dp), parameter :: L_SUN_CGS    = 3.828d33       ! solar luminosity erg/s
   real(dp), parameter :: SIGMA_SB     = 5.6704d-5      ! Stefan-Boltzmann erg/s/cm^2/K^4
   real(dp), parameter :: TINY_VALUE   = 1.0d-10

   public :: dp
   public :: cc_find_interval
   public :: cc_find_containing_cell
   public :: cc_find_nearest_point
   public :: cc_hermite_interp_vector
   public :: cc_trilinear_interp_vector
   public :: cc_dilute_flux
   public :: cc_simpson_integration
   public :: cc_trapz_integration
   public :: cc_interp_filter_onto_sed
   public :: cc_synthetic_flux
   public :: cc_magnitude
   public :: cc_bolometric_flux
   public :: cc_bolometric_magnitude
   public :: cc_vega_zero_point
   public :: cc_ab_zero_point
   public :: cc_st_zero_point

contains

   ! =========================================================================
   ! Grid location utilities
   ! =========================================================================

   subroutine cc_find_interval(x, n, val, i_out, t_out)
      ! Locate the interval in sorted array x(1:n) that contains val.
      ! Returns i_out such that x(i_out) <= val <= x(i_out+1),
      ! and t_out = (val - x(i)) / (x(i+1) - x(i)) in [0,1].
      ! Clamped to [1, n-1].
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: x(n), val
      integer,  intent(out) :: i_out
      real(dp), intent(out) :: t_out

      integer  :: lo, hi, mid
      real(dp) :: denom

      if (n <= 1) then
         i_out = 1; t_out = 0.0_dp; return
      end if

      ! Binary search
      lo = 1; hi = n
      do while (hi - lo > 1)
         mid = (lo + hi) / 2
         if (val >= x(mid)) then
            lo = mid
         else
            hi = mid
         end if
      end do

      ! Clamp
      i_out = max(1, min(lo, n - 1))

      denom = x(i_out + 1) - x(i_out)
      if (abs(denom) > 0.0_dp) then
         t_out = (val - x(i_out)) / denom
      else
         t_out = 0.0_dp
      end if
      t_out = max(0.0_dp, min(1.0_dp, t_out))
   end subroutine cc_find_interval


   subroutine cc_find_containing_cell(x_val, y_val, z_val, &
                                      x_grid, nx, y_grid, ny, z_grid, nz, &
                                      i_x, i_y, i_z, t_x, t_y, t_z)
      integer,  intent(in)  :: nx, ny, nz
      real(dp), intent(in)  :: x_val, y_val, z_val
      real(dp), intent(in)  :: x_grid(nx), y_grid(ny), z_grid(nz)
      integer,  intent(out) :: i_x, i_y, i_z
      real(dp), intent(out) :: t_x, t_y, t_z

      call cc_find_interval(x_grid, nx, x_val, i_x, t_x)
      call cc_find_interval(y_grid, ny, y_val, i_y, t_y)
      call cc_find_interval(z_grid, nz, z_val, i_z, t_z)
   end subroutine cc_find_containing_cell


   subroutine cc_find_nearest_point(x_val, y_val, z_val, &
                                    x_grid, nx, y_grid, ny, z_grid, nz, &
                                    i_x, i_y, i_z)
      integer,  intent(in)  :: nx, ny, nz
      real(dp), intent(in)  :: x_val, y_val, z_val
      real(dp), intent(in)  :: x_grid(nx), y_grid(ny), z_grid(nz)
      integer,  intent(out) :: i_x, i_y, i_z

      integer  :: k
      real(dp) :: best, d

      best = abs(x_grid(1) - x_val); i_x = 1
      do k = 2, nx
         d = abs(x_grid(k) - x_val)
         if (d < best) then; best = d; i_x = k; end if
      end do

      best = abs(y_grid(1) - y_val); i_y = 1
      do k = 2, ny
         d = abs(y_grid(k) - y_val)
         if (d < best) then; best = d; i_y = k; end if
      end do

      best = abs(z_grid(1) - z_val); i_z = 1
      do k = 2, nz
         d = abs(z_grid(k) - z_val)
         if (d < best) then; best = d; i_z = k; end if
      end do
   end subroutine cc_find_nearest_point


   ! =========================================================================
   ! Hermite basis functions
   ! =========================================================================

   pure real(dp) function h00(t)
      real(dp), intent(in) :: t
      h00 = (1.0_dp + 2.0_dp*t) * (1.0_dp - t)**2
   end function h00

   pure real(dp) function h10(t)
      real(dp), intent(in) :: t
      h10 = t * (1.0_dp - t)**2
   end function h10

   pure real(dp) function h01(t)
      real(dp), intent(in) :: t
      h01 = t**2 * (3.0_dp - 2.0_dp*t)
   end function h01

   pure real(dp) function h11(t)
      real(dp), intent(in) :: t
      h11 = t**2 * (t - 1.0_dp)
   end function h11


   subroutine compute_derivatives_4d(f, nt, nl, nm, nw, &
                                     i, j, k, lam, &
                                     teff_grid, logg_grid, meta_grid, &
                                     df_dx, df_dy, df_dz)
      ! Central / one-sided finite differences for Hermite tangents.
      integer,  intent(in)  :: nt, nl, nm, nw
      real(dp), intent(in)  :: f(nt, nl, nm, nw)
      integer,  intent(in)  :: i, j, k, lam
      real(dp), intent(in)  :: teff_grid(nt), logg_grid(nl), meta_grid(nm)
      real(dp), intent(out) :: df_dx, df_dy, df_dz

      ! x (Teff)
      if (nt == 1) then
         df_dx = 0.0_dp
      else if (i > 1 .and. i < nt) then
         df_dx = (f(i+1,j,k,lam) - f(i-1,j,k,lam)) / (teff_grid(i+1) - teff_grid(i-1))
      else if (i == 1) then
         df_dx = (f(i+1,j,k,lam) - f(i,j,k,lam))   / (teff_grid(i+1) - teff_grid(i))
      else
         df_dx = (f(i,j,k,lam)   - f(i-1,j,k,lam)) / (teff_grid(i)   - teff_grid(i-1))
      end if

      ! y (logg)
      if (nl == 1) then
         df_dy = 0.0_dp
      else if (j > 1 .and. j < nl) then
         df_dy = (f(i,j+1,k,lam) - f(i,j-1,k,lam)) / (logg_grid(j+1) - logg_grid(j-1))
      else if (j == 1) then
         df_dy = (f(i,j+1,k,lam) - f(i,j,k,lam))   / (logg_grid(j+1) - logg_grid(j))
      else
         df_dy = (f(i,j,k,lam)   - f(i,j-1,k,lam)) / (logg_grid(j)   - logg_grid(j-1))
      end if

      ! z ([M/H])
      if (nm == 1) then
         df_dz = 0.0_dp
      else if (k > 1 .and. k < nm) then
         df_dz = (f(i,j,k+1,lam) - f(i,j,k-1,lam)) / (meta_grid(k+1) - meta_grid(k-1))
      else if (k == 1) then
         df_dz = (f(i,j,k+1,lam) - f(i,j,k,lam))   / (meta_grid(k+1) - meta_grid(k))
      else
         df_dz = (f(i,j,k,lam)   - f(i,j,k-1,lam)) / (meta_grid(k)   - meta_grid(k-1))
      end if
   end subroutine compute_derivatives_4d


   ! =========================================================================
   ! Hermite tensor interpolation (vectorised over wavelength)
   ! =========================================================================

   subroutine cc_hermite_interp_vector(teff, logg, meta, &
                                       teff_grid, nt, &
                                       logg_grid, nl, &
                                       meta_grid, nm, &
                                       flux_cube, nw, &
                                       result_flux, ierr)
      ! Interpolate the full SED at (teff, logg, meta) from flux_cube.
      ! Cell location is computed once; basis functions are precomputed;
      ! the wavelength loop only does the weighted sum.
      !
      ! flux_cube has shape (nt, nl, nm, nw) in memory.
      integer,  intent(in)  :: nt, nl, nm, nw
      real(dp), intent(in)  :: teff, logg, meta
      real(dp), intent(in)  :: teff_grid(nt), logg_grid(nl), meta_grid(nm)
      real(dp), intent(in)  :: flux_cube(nt, nl, nm, nw)
      real(dp), intent(out) :: result_flux(nw)
      integer,  intent(out) :: ierr

      integer  :: i_x, i_y, i_z, ix, iy, iz, lam
      real(dp) :: t_x, t_y, t_z, dx, dy, dz, f_sum
      real(dp) :: df_dx, df_dy, df_dz
      real(dp) :: h_x(2), hx_d(2), h_y(2), hy_d(2), h_z(2), hz_d(2)

      ierr = 0

      call cc_find_containing_cell(teff, logg, meta, &
                                   teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                   i_x, i_y, i_z, t_x, t_y, t_z)

      ! Outside grid: fall back to nearest point
      if (i_x < 1 .or. i_x >= nt .or. &
          i_y < 1 .or. i_y >= nl .or. &
          i_z < 1 .or. i_z >= nm) then
         call cc_find_nearest_point(teff, logg, meta, &
                                    teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                    i_x, i_y, i_z)
         result_flux = flux_cube(i_x, i_y, i_z, :)
         ierr = 1   ! flag: clamped to boundary
         return
      end if

      dx = teff_grid(i_x + 1) - teff_grid(i_x)
      dy = logg_grid(i_y + 1) - logg_grid(i_y)
      dz = meta_grid(i_z + 1) - meta_grid(i_z)

      ! Precompute Hermite basis (same for all wavelengths)
      h_x  = [h00(t_x), h01(t_x)]
      hx_d = [h10(t_x), h11(t_x)]
      h_y  = [h00(t_y), h01(t_y)]
      hy_d = [h10(t_y), h11(t_y)]
      h_z  = [h00(t_z), h01(t_z)]
      hz_d = [h10(t_z), h11(t_z)]

      ! Wavelength loop: only the derivative reads and sum vary
      do lam = 1, nw
         f_sum = 0.0_dp
         do iz = 0, 1
            do iy = 0, 1
               do ix = 0, 1
                  call compute_derivatives_4d( &
                     flux_cube, nt, nl, nm, nw, &
                     i_x+ix, i_y+iy, i_z+iz, lam, &
                     teff_grid, logg_grid, meta_grid, &
                     df_dx, df_dy, df_dz)

                  f_sum = f_sum &
                     + h_x(ix+1)  * h_y(iy+1)  * h_z(iz+1)  * flux_cube(i_x+ix, i_y+iy, i_z+iz, lam) &
                     + hx_d(ix+1) * h_y(iy+1)  * h_z(iz+1)  * dx * df_dx &
                     + h_x(ix+1)  * hy_d(iy+1) * h_z(iz+1)  * dy * df_dy &
                     + h_x(ix+1)  * h_y(iy+1)  * hz_d(iz+1) * dz * df_dz
               end do
            end do
         end do
         result_flux(lam) = max(TINY_VALUE, f_sum)
      end do
   end subroutine cc_hermite_interp_vector


   ! =========================================================================
   ! Trilinear interpolation (vectorised over wavelength)
   ! =========================================================================

   subroutine cc_trilinear_interp_vector(teff, logg, meta, &
                                         teff_grid, nt, &
                                         logg_grid, nl, &
                                         meta_grid, nm, &
                                         flux_cube, nw, &
                                         result_flux, ierr)
      integer,  intent(in)  :: nt, nl, nm, nw
      real(dp), intent(in)  :: teff, logg, meta
      real(dp), intent(in)  :: teff_grid(nt), logg_grid(nl), meta_grid(nm)
      real(dp), intent(in)  :: flux_cube(nt, nl, nm, nw)
      real(dp), intent(out) :: result_flux(nw)
      integer,  intent(out) :: ierr

      integer  :: i_x, i_y, i_z, lam
      real(dp) :: t_x, t_y, t_z
      real(dp) :: c000, c001, c010, c011, c100, c101, c110, c111
      real(dp) :: c00, c01, c10, c11, c0, c1, lin_val

      ierr = 0

      call cc_find_containing_cell(teff, logg, meta, &
                                   teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                   i_x, i_y, i_z, t_x, t_y, t_z)

      if (i_x < 1 .or. i_x >= nt .or. &
          i_y < 1 .or. i_y >= nl .or. &
          i_z < 1 .or. i_z >= nm) then
         call cc_find_nearest_point(teff, logg, meta, &
                                    teff_grid, nt, logg_grid, nl, meta_grid, nm, &
                                    i_x, i_y, i_z)
         result_flux = flux_cube(i_x, i_y, i_z, :)
         ierr = 1
         return
      end if

      do lam = 1, nw
         c000 = flux_cube(i_x,   i_y,   i_z,   lam)
         c100 = flux_cube(i_x+1, i_y,   i_z,   lam)
         c010 = flux_cube(i_x,   i_y+1, i_z,   lam)
         c110 = flux_cube(i_x+1, i_y+1, i_z,   lam)
         c001 = flux_cube(i_x,   i_y,   i_z+1, lam)
         c101 = flux_cube(i_x+1, i_y,   i_z+1, lam)
         c011 = flux_cube(i_x,   i_y+1, i_z+1, lam)
         c111 = flux_cube(i_x+1, i_y+1, i_z+1, lam)

         c00 = c000*(1.0_dp - t_x) + c100*t_x
         c01 = c001*(1.0_dp - t_x) + c101*t_x
         c10 = c010*(1.0_dp - t_x) + c110*t_x
         c11 = c011*(1.0_dp - t_x) + c111*t_x

         c0 = c00*(1.0_dp - t_y) + c10*t_y
         c1 = c01*(1.0_dp - t_y) + c11*t_y

         lin_val = c0*(1.0_dp - t_z) + c1*t_z
         result_flux(lam) = max(TINY_VALUE, lin_val)
      end do
   end subroutine cc_trilinear_interp_vector


   ! =========================================================================
   ! Flux dilution
   ! =========================================================================

   subroutine cc_dilute_flux(surface_flux, nw, R, d, observed_flux)
      ! F_obs(λ) = F_surf(λ) × (R / d)²
      integer,  intent(in)  :: nw
      real(dp), intent(in)  :: surface_flux(nw), R, d
      real(dp), intent(out) :: observed_flux(nw)
      observed_flux = surface_flux * (R / d)**2
   end subroutine cc_dilute_flux


   ! =========================================================================
   ! Numerical integration
   ! =========================================================================

   subroutine cc_trapz_integration(x, y, n, result)
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: x(n), y(n)
      real(dp), intent(out) :: result
      integer :: i
      result = 0.0_dp
      do i = 1, n - 1
         result = result + 0.5_dp * (y(i) + y(i+1)) * (x(i+1) - x(i))
      end do
   end subroutine cc_trapz_integration


   subroutine cc_simpson_integration(x, y, n, result)
      ! Adaptive Simpson / trapezoid fallback (mirrors colors_utils.f90).
      ! Uses composite Simpson on odd-length arrays, trapezoid on even.
      integer,  intent(in)  :: n
      real(dp), intent(in)  :: x(n), y(n)
      real(dp), intent(out) :: result

      integer  :: i
      real(dp) :: h, s

      if (n < 2) then
         result = 0.0_dp; return
      end if

      if (mod(n, 2) == 1) then
         ! odd n: composite Simpson 1/3
         s = 0.0_dp
         do i = 1, n - 2, 2
            h = x(i+2) - x(i)
            s = s + h / 6.0_dp * (y(i) + 4.0_dp*y(i+1) + y(i+2))
         end do
         result = s
      else
         ! even n: trapezoid fallback
         call cc_trapz_integration(x, y, n, result)
      end if
   end subroutine cc_simpson_integration


   ! =========================================================================
   ! Filter interpolation
   ! =========================================================================

   subroutine cc_interp_filter_onto_sed(filt_wave, filt_trans, nf, &
                                         sed_wave, nw, &
                                         filt_on_sed, ierr)
      ! Linear interpolation of filter transmission onto the SED wavelength grid.
      ! Clamps to 0 outside the filter wavelength range.
      integer,  intent(in)  :: nf, nw
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(in)  :: sed_wave(nw)
      real(dp), intent(out) :: filt_on_sed(nw)
      integer,  intent(out) :: ierr

      integer  :: i, lo, hi, mid
      real(dp) :: t, denom

      ierr = 0
      do i = 1, nw
         if (sed_wave(i) <= filt_wave(1)) then
            filt_on_sed(i) = 0.0_dp
         else if (sed_wave(i) >= filt_wave(nf)) then
            filt_on_sed(i) = 0.0_dp
         else
            lo = 1; hi = nf
            do while (hi - lo > 1)
               mid = (lo + hi) / 2
               if (sed_wave(i) >= filt_wave(mid)) then
                  lo = mid
               else
                  hi = mid
               end if
            end do
            denom = filt_wave(lo+1) - filt_wave(lo)
            if (abs(denom) > 0.0_dp) then
               t = (sed_wave(i) - filt_wave(lo)) / denom
            else
               t = 0.0_dp
            end if
            filt_on_sed(i) = filt_trans(lo) + t*(filt_trans(lo+1) - filt_trans(lo))
            filt_on_sed(i) = max(0.0_dp, filt_on_sed(i))
         end if
      end do
   end subroutine cc_interp_filter_onto_sed


   ! =========================================================================
   ! Synthetic photometry
   ! =========================================================================

   subroutine cc_synthetic_flux(sed_wave, obs_flux, filt_on_sed_grid, nw, &
                                 synthetic_flux, ierr)
      ! Photon-counting in-band flux:
      !   F_band = ∫ F(λ) T(λ) λ dλ / ∫ T(λ) λ dλ
      integer,  intent(in)  :: nw
      real(dp), intent(in)  :: sed_wave(nw), obs_flux(nw), filt_on_sed_grid(nw)
      real(dp), intent(out) :: synthetic_flux
      integer,  intent(out) :: ierr

      real(dp) :: num, den
      real(dp), dimension(nw) :: integrand_num, integrand_den

      ierr = 0
      integrand_num = obs_flux * filt_on_sed_grid * sed_wave
      integrand_den = filt_on_sed_grid * sed_wave

      call cc_simpson_integration(sed_wave, integrand_num, nw, num)
      call cc_simpson_integration(sed_wave, integrand_den, nw, den)

      if (den > 0.0_dp) then
         synthetic_flux = num / den
      else
         synthetic_flux = -1.0_dp
         ierr = 1
      end if
   end subroutine cc_synthetic_flux


   subroutine cc_magnitude(band_flux, zero_point, mag, ierr)
      ! m = -2.5 log10(F_band / F_zp)
      real(dp), intent(in)  :: band_flux, zero_point
      real(dp), intent(out) :: mag
      integer,  intent(out) :: ierr

      ierr = 0
      if (band_flux > 0.0_dp .and. zero_point > 0.0_dp) then
         mag = -2.5_dp * log10(band_flux / zero_point)
      else
         mag = -99.9_dp
         ierr = 1
      end if
   end subroutine cc_magnitude


   ! =========================================================================
   ! Bolometric quantities
   ! =========================================================================

   subroutine cc_bolometric_flux(sed_wave, obs_flux, nw, bol_flux, ierr)
      ! Integrate the diluted SED over all wavelengths.
      integer,  intent(in)  :: nw
      real(dp), intent(in)  :: sed_wave(nw), obs_flux(nw)
      real(dp), intent(out) :: bol_flux
      integer,  intent(out) :: ierr
      ierr = 0
      call cc_simpson_integration(sed_wave, obs_flux, nw, bol_flux)
      if (bol_flux <= 0.0_dp) ierr = 1
   end subroutine cc_bolometric_flux


   subroutine cc_bolometric_magnitude(bol_flux, bol_mag, ierr)
      ! M_bol = M_bol_sun - 2.5 log10(F_bol / F_bol_sun_at_10pc)
      ! where F_bol_sun_at_10pc = L_sun / (4 pi (10 pc)^2)
      ! We use the IAU 2015 relation: M_bol = -2.5 log10(F_bol) - 18.997350
      ! (calibrated so the Sun has M_bol = 4.74 at 10 pc)
      real(dp), intent(in)  :: bol_flux
      real(dp), intent(out) :: bol_mag
      integer,  intent(out) :: ierr

      real(dp), parameter :: ZP = 18.997350d0  ! IAU 2015 bolometric flux zero-point

      ierr = 0
      if (bol_flux > 0.0_dp) then
         bol_mag = -2.5_dp * log10(bol_flux) - ZP
      else
         bol_mag = -99.9_dp
         ierr = 1
      end if
   end subroutine cc_bolometric_magnitude


   ! =========================================================================
   ! Zero-point computation  (called once at init from Python)
   ! =========================================================================

   subroutine cc_vega_zero_point(vega_wave, vega_flux, nv, &
                                  filt_wave, filt_trans, nf, &
                                  zp, ierr)
      integer,  intent(in)  :: nv, nf
      real(dp), intent(in)  :: vega_wave(nv), vega_flux(nv)
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr

      real(dp), dimension(nv) :: trans_on_vega, conv_flux
      real(dp) :: num, den

      ierr = 0
      call cc_interp_filter_onto_sed(filt_wave, filt_trans, nf, &
                                      vega_wave, nv, trans_on_vega, ierr)
      conv_flux = vega_flux * trans_on_vega * vega_wave

      call cc_simpson_integration(vega_wave, conv_flux, nv, num)
      call cc_simpson_integration(vega_wave, trans_on_vega * vega_wave, nv, den)

      if (den > 0.0_dp) then
         zp = num / den
      else
         zp = -1.0_dp; ierr = 1
      end if
   end subroutine cc_vega_zero_point


   subroutine cc_ab_zero_point(filt_wave, filt_trans, nf, zp, ierr)
      integer,  intent(in)  :: nf
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr

      integer  :: i
      real(dp), dimension(nf) :: f_ab
      real(dp) :: num, den

      ierr = 0
      do i = 1, nf
         if (filt_wave(i) > 0.0_dp) then
            f_ab(i) = AB_FNU_ZP * (CLIGHT_CM_S * 1.0d8) / filt_wave(i)**2
         else
            f_ab(i) = 0.0_dp
         end if
      end do

      call cc_simpson_integration(filt_wave, f_ab * filt_trans * filt_wave, nf, num)
      call cc_simpson_integration(filt_wave, filt_trans * filt_wave, nf, den)

      if (den > 0.0_dp) then
         zp = num / den
      else
         zp = -1.0_dp; ierr = 1
      end if
   end subroutine cc_ab_zero_point


   subroutine cc_st_zero_point(filt_wave, filt_trans, nf, zp, ierr)
      integer,  intent(in)  :: nf
      real(dp), intent(in)  :: filt_wave(nf), filt_trans(nf)
      real(dp), intent(out) :: zp
      integer,  intent(out) :: ierr

      real(dp), dimension(nf) :: f_st
      real(dp) :: num, den

      ierr = 0
      f_st = ST_FLAM_ZP

      call cc_simpson_integration(filt_wave, f_st * filt_trans * filt_wave, nf, num)
      call cc_simpson_integration(filt_wave, filt_trans * filt_wave, nf, den)

      if (den > 0.0_dp) then
         zp = num / den
      else
         zp = -1.0_dp; ierr = 1
      end if
   end subroutine cc_st_zero_point

end module cc_kernels
