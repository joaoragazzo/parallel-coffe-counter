from mpi4py import MPI
import math, os, cv2, pickle
import customtkinter as ctk
from roles.worker import Worker
from typing import override
from PIL import Image, ImageTk

class Master(Worker):
    
    def __init__(self, comm: MPI.Comm, **kwargs):
        self.comm = comm
        self.cap = cv2.VideoCapture(0)
        self.size = comm.Get_size()
        self.n = int(math.sqrt(self.size - 2))
        
        self.parameters = {
            'area_threshold_X': 500,
        }

        if self.n * self.n != self.size - 2:
            print("O número de processos escravos (size - 2) deve ser um quadrado perfeito.")
            self.comm.Abort()

    def divide_image(self, frame):
        height, width, _ = frame.shape
        h_step = height // self.n
        w_step = width // self.n

        quadrants = []
        positions = []
        for i in range(self.n):
            for j in range(self.n):
                y_start = i * h_step
                y_end = (i + 1) * h_step if i != self.n - 1 else height
                x_start = j * w_step
                x_end = (j + 1) * w_step if j != self.n - 1 else width
                quadrant_data = frame[y_start:y_end, x_start:x_end]
                quadrant_position = (i, j)
                quadrants.append((quadrant_position, quadrant_data))
                positions.append((x_start, y_start))
        return quadrants, positions, h_step, w_step

    def create_gui(self):
        self.root = ctk.CTk()
        self.root.title("Master controller")

        self.slider_vars = {}
        self.value_labels_vars = {}

        row = 0
        for param_name in self.parameters:
            self.slider_vars[param_name] = ctk.IntVar(value=self.parameters[param_name])
            self.value_labels_vars[param_name] = ctk.StringVar()
            self.value_labels_vars[param_name].set(f"{self.parameters[param_name]}")

            label = ctk.CTkLabel(self.root, text=param_name)
            slider = ctk.CTkSlider(self.root, from_=0, to=1000, variable=self.slider_vars[param_name])
            value_label = ctk.CTkLabel(self.root, textvariable=self.value_labels_vars[param_name])

            def update_value_label(var=param_name):
                value = self.slider_vars[var].get()
                self.value_labels_vars[var].set(f"{value}")

            self.slider_vars[param_name].trace_add('write', lambda *args, var=param_name: update_value_label(var))

            label.grid(row=row, column=0, padx=5, pady=5, sticky='w')
            slider.grid(row=row, column=1, padx=5, pady=5, sticky='we')
            value_label.grid(row=row, column=2, padx=5, pady=5, sticky='e')

            row += 1

        self.root.columnconfigure(1, weight=1)

    def update_parameters_from_gui(self):
        for param_name in self.parameters:
            self.parameters[param_name] = self.slider_vars[param_name].get()

    def create_classification_gui(self, bean_images):
        self.classification_gui_open = True

        self.classification_window = ctk.CTkToplevel(self.root)
        self.classification_window.title("Bean Classification")
        self.classification_window.protocol("WM_DELETE_WINDOW", self.on_classification_gui_close)  # Handle window close

        self.bean_images_iter = iter(bean_images)
        self.current_bean_image = None

        self.bean_image_label = ctk.CTkLabel(self.classification_window)
        self.bean_image_label.pack()

        button_frame = ctk.CTkFrame(self.classification_window)
        button_frame.pack(pady=10)

        btn_green = ctk.CTkButton(button_frame, text="Green", command=lambda: self.classify_bean("green"))
        btn_white = ctk.CTkButton(button_frame, text="White", command=lambda: self.classify_bean("white"))
        btn_roasted = ctk.CTkButton(button_frame, text="Roasted", command=lambda: self.classify_bean("roasted"))
        btn_black = ctk.CTkButton(button_frame, text="Black", command=lambda: self.classify_bean("black"))
        btn_invalidate = ctk.CTkButton(button_frame, text="Invalidate", command=self.invalidate_bean)

        btn_green.grid(row=0, column=0, padx=5)
        btn_white.grid(row=0, column=1, padx=5)
        btn_roasted.grid(row=0, column=2, padx=5)
        btn_black.grid(row=0, column=3, padx=5)
        btn_invalidate.grid(row=0, column=4, padx=5)

        self.display_next_bean()

    def on_classification_gui_close(self):
        self.classification_gui_open = False
        self.classification_window.destroy()

    def display_next_bean(self):
        try:
            self.current_bean_image = next(self.bean_images_iter)
            img_rgb = cv2.cvtColor(self.current_bean_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            desired_width = 300  
            desired_height = 300  
            
            img_tk = ctk.CTkImage(img_pil, size=(desired_width, desired_height))
            
            self.bean_image_label.configure(image=img_tk)
            self.bean_image_label.image = img_tk  
        except StopIteration:
            self.current_bean_image = None
            self.bean_image_label.configure(image=None)
            self.bean_image_label.image = None
            print("All beans in this batch have been classified.")
            self.on_classification_gui_close()




    def invalidate_bean(self):
        if self.current_bean_image is not None:
            print("Bean invalidated and not saved.")

        self.display_next_bean()

    def classify_bean(self, bean_type):
        if self.current_bean_image is not None:
            folder_path = os.path.join("beans_dataset", bean_type)
            os.makedirs(folder_path, exist_ok=True)
            image_count = len(os.listdir(folder_path))
            image_path = os.path.join(folder_path, f"{bean_type}_{image_count+1}.png")
            cv2.imwrite(image_path, self.current_bean_image)
            print(f"Bean saved to {image_path}")

        self.display_next_bean()


    def main(self):
        self.bean_images_queue = []  
        self.classification_gui_open = False  
        MAX_IMAGES = 10

        print("webcam ativa")

        if not self.cap.isOpened():
            print("Não foi possível abrir a webcam.")
            self.comm.Abort()

        # Create the GUI
        self.create_gui()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Falha ao capturar o frame.")
                    break

                self.update_parameters_from_gui()

                quadrants, positions, h_step, w_step = self.divide_image(frame)

                for idx, ((quadrant_position, quadrant_data), (x_start, y_start)) in enumerate(zip(quadrants, positions)):
                    dest_rank = idx + 2  
                    data_to_send = (quadrant_position, quadrant_data, self.parameters)
                    self.comm.send(data_to_send, dest=dest_rank, tag=11)

                somatory = 0

                results = {}
                for i in range(2, self.size):
                    serialized_data = self.comm.recv(source=i, tag=22)
                    quadrant_position, result = pickle.loads(serialized_data)
                    results[quadrant_position] = result
                    somatory += result['count']
                print(f"Grãos de café contados: {str(somatory)}")

                all_bean_images = []
                for idx, ((quadrant_position, _), (x_offset, y_offset)) in enumerate(zip(quadrants, positions)):
                    if quadrant_position in results:
                        result = results[quadrant_position]
                        bean_images = result.get('bean_images', [])
                        all_bean_images.extend(bean_images)

                available_space = 5 - len(self.bean_images_queue)
                if available_space > 0:
                    images_to_add = all_bean_images[:available_space]
                    self.bean_images_queue.extend(images_to_add)
                    discarded_count = len(all_bean_images) - len(images_to_add)
                    if discarded_count > 0:
                        print(f"Queue full. Discarded {discarded_count} bean images.")
                else:
                    if len(all_bean_images) > 0:
                        print(f"Queue full. Discarded {len(all_bean_images)} bean images.")

                if not self.classification_gui_open and self.bean_images_queue:
                    images_to_classify = self.bean_images_queue[:MAX_IMAGES]
                    self.bean_images_queue = self.bean_images_queue[MAX_IMAGES:]
                    self.create_classification_gui(images_to_classify)

                height, width, _ = frame.shape
                for i in range(1, self.n):
                    y = i * h_step
                    cv2.line(frame, (0, y), (width, y), (0, 255, 0), 1)
                for j in range(1, self.n):
                    x = j * w_step
                    cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 1)

                for idx, ((quadrant_position, _), (x_offset, y_offset)) in enumerate(zip(quadrants, positions)):
                    if quadrant_position in results:
                        result = results[quadrant_position]
                        ellipses = result.get('ellipses', [])

                        for ellipse in ellipses:
                            center = (
                                int(ellipse['center'][0] + x_offset),
                                int(ellipse['center'][1] + y_offset)
                            )
                            axes = (
                                int(ellipse['axes'][0] / 2),
                                int(ellipse['axes'][1] / 2)
                            )
                            angle = ellipse['angle']

                            cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 0, 255), 4)

                cv2.imshow('Webcam', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    for i in range(2, self.size):
                        self.comm.send(None, dest=i, tag=99)
                    break

                self.root.update()

        except KeyboardInterrupt:
            print("Interrupção pelo usuário.")
            for i in range(2, self.size):
                self.comm.send(None, dest=i, tag=99)
            self.comm.Abort()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()

    @override
    def work(self):
        self.main()
        pass
