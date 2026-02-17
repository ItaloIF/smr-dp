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