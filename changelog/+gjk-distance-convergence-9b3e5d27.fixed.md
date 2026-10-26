Fix convex contacts being dropped when their true separation is just inside the contact gap; GJK now converges distances to float32 rounding instead of a 1e-4 relative tolerance.
