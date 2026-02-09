program record_processing
    implicit none
    integer :: i, j, k
    ! Test the smooth_fourier_amplitude subroutine
    integer :: n
    real(4), allocatable :: freq(:), fas(:), smooth_fas(:)
    real(4) :: d

    n = 2 ** 17
    d = 0.5d0
    allocate(freq(n), fas(n), smooth_fas(n))

    ! Initialize test data
    freq(1:5) = (/ 1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0 /)
    fas(1:5) = (/ 10.0d0, 20.0d0, 30.0d0, 40.0d0, 50.0d0 /)
    freq(6:n) = 0.0d0
    fas(6:n) = 0.0d0

    call smooth_fourier_amplitude(n, freq, fas, d, smooth_fas)

    deallocate(freq, fas, smooth_fas)

end program record_processing

subroutine integration_linear_acceleration(n, dt, a, v, d)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: dt
    real(4), intent(in) :: a(n)
    real(4), intent(out) :: v(n)
    real(4), intent(out) :: d(n)

    integer :: i

    v(1) = 0.0d0
    d(1) = 0.0d0

    do i = 2, n
        v(i) = v(i-1) + (a(i) + a(i-1)) / 2 * dt
        d(i) = d(i-1) + v(i-1) * dt + (1.0d0/6.0d0) * (2.0d0*a(i-1) + a(i)) * dt**2
    end do

end subroutine integration_linear_acceleration

subroutine differentiate_displacement(n, dt, dis, vel, acc)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: dt
    real(4), intent(in) :: dis(n)
    real(4), intent(out) :: vel(n)
    real(4), intent(out) :: acc(n)

    integer :: i
    real(4) :: dt2
    dt2 = dt * dt

    ! Velocity using central differences
    do i = 2, n-1
        vel(i) = (dis(i+1) - dis(i-1)) / (2 * dt)
    end do
    vel(1) = (dis(2) - dis(1)) / dt          ! forward difference
    vel(n) = (dis(n) - dis(n-1)) / dt       ! backward difference

    ! Acceleration using central differences
    do i = 2, n-1
        acc(i) = (dis(i+1) - 2*dis(i) + dis(i-1)) / (dt2)
    end do
    acc(1) = (dis(3) - 2*dis(2) + dis(1)) / (dt2)   ! approx forward
    acc(n) = (dis(n) - 2*dis(n-1) + dis(n-2)) / (dt2)  ! approx backward

end subroutine differentiate_displacement

subroutine rotd50(n, dt, a, nbin, pga)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: dt
    real(4), intent(in) :: a(n)
    integer, intent(in) :: nbin
    real(4), intent(out) :: pga


    integer :: i
    real(4) :: t, t0, t1
    real(4) :: a_max, a_min
    real(4) :: delta_angle


    delta_angle = 180.0d0 / nbin


end subroutine rotd50

subroutine smooth_fourier_amplitude(n, freq, fas, d, smooth_fas)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: freq(n)
    real(4), intent(in) :: fas(n)
    real(4), intent(in) :: d
    real(4), intent(out) :: smooth_fas(n)

    integer :: i, j, n_w
    real(4) :: sum
    real(4) :: lower_bound, upper_bound

    do i = 1, n
        sum = 0.0d0
        lower_bound = freq(i) * 10.0d0 ** (-d/2.0d0)
        upper_bound = freq(i) * 10.0d0 ** (d/2.0d0)
        n_w = 0
        do j = 1, n
            if (freq(j) >= lower_bound .and. freq(j) <= upper_bound) then
                sum = sum + log(fas(j))
                n_w = n_w + 1
            end if
        end do
        smooth_fas(i) = exp(sum / real(n_w, 4))
    end do

end subroutine smooth_fourier_amplitude