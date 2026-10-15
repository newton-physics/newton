Fix GJK distance queries reporting positive convex gaps below the convergence tolerance as overlap with a zero normal; they now return the gap and an oriented unit normal.
