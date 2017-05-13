import snn1337
class TestSNN1337(unittest.TestCase):

    def set_up(self):
        self.hello_message = "Hello, snn1337!"

    def test_print_hello(self):
        assert(self.hello_message)

