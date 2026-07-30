raw_probe <- function() {
  .Call("RawScalar_R", PACKAGE = "libprobe")
}
