from manim import *
import numpy as np

class DetailedModelAnimation(Scene):
    def construct(self):
        # --- 1. TITLE ---
        title = Text("SRS Matrix Prediction", font_size=32).to_edge(UP)
        self.play(Write(title))

        # --- HELPER FUNCTIONS ---
        
        def get_conv_stack(num_maps, color, label="Conv2D"):
            stack = VGroup()
            for i in range(num_maps):
                rect = Rectangle(height=1.2, width=1.0, fill_color=color, fill_opacity=0.4, stroke_width=1)
                rect.shift(RIGHT * i * 0.12 + UP * i * 0.12)
                stack.add(rect)
            lbl = Text(label, font_size=10).next_to(stack, DOWN, buff=0.1)
            return VGroup(stack, lbl)

        def get_partial_dense(color, label="Dense"):
            neurons_top = VGroup(*[Dot(radius=0.08, color=color) for _ in range(3)]).arrange(DOWN, buff=0.15)
            dots = VGroup(*[Dot(radius=0.02, color=GRAY) for _ in range(3)]).arrange(DOWN, buff=0.1).next_to(neurons_top, DOWN, buff=0.15)
            neurons_bottom = VGroup(*[Dot(radius=0.08, color=color) for _ in range(3)]).arrange(DOWN, buff=0.15).next_to(dots, DOWN, buff=0.15)
            layer = VGroup(neurons_top, dots, neurons_bottom)
            lbl = Text(label, font_size=11).next_to(layer, UP, buff=0.2)
            return VGroup(layer, lbl)

        # --- STAGE 1: INPUTS ---
        radio_in = Text("Radio (E2)", font_size=16, color=BLUE).to_edge(LEFT).shift(UP*1.8)
        srs_in = Text("SRS (Raw)", font_size=16, color=GREEN).to_edge(LEFT).shift(DOWN*1.8)
        
        r_conv = get_conv_stack(5, BLUE_E, "Conv 128").next_to(radio_in, RIGHT, buff=0.6)
        s_res = RoundedRectangle(height=2.0, width=2.2, color=GREEN_B, fill_opacity=0.1).next_to(srs_in, RIGHT, buff=3.0)
        s_res_lbl = Text("ResNet Block", font_size=11).next_to(s_res, UP, buff=0.1)

        self.play(Create(radio_in), Create(srs_in), Create(r_conv), Create(s_res), Write(s_res_lbl))

        # --- STAGE 2: FUSION & LSTM ---
        fusion_box = Rectangle(height=2.8, width=0.6, color=PURPLE, fill_opacity=0.6).shift(RIGHT*0.5)
        lstm_box = Square(side_length=1.4, color=ORANGE, fill_opacity=0.3).next_to(fusion_box, RIGHT, buff=0.8)
        lstm_lbl = Text("LSTM", font_size=14).move_to(lstm_box)

        self.play(
            r_conv.animate.next_to(fusion_box, LEFT, buff=0.2),
            s_res.animate.next_to(fusion_box, LEFT, buff=0.2).scale(0.6),
            Create(fusion_box),
            Create(lstm_box), Write(lstm_lbl)
        )

        # --- STAGE 3: DENSE LAYERS ---
        self.play(FadeOut(r_conv), FadeOut(s_res), FadeOut(fusion_box), FadeOut(s_res_lbl))
        self.play(lstm_box.animate.to_edge(LEFT).shift(RIGHT*0.5))
        lstm_lbl.add_updater(lambda m: m.move_to(lstm_box.get_center()))

        d1 = get_partial_dense(RED, "Dense 1024").shift(RIGHT*0.5)
        d2 = get_partial_dense(RED_E, "Dense 2048").shift(RIGHT*2.5)
        out_layer = get_partial_dense(GOLD, "Dense Layer").shift(RIGHT*4.5)

        self.play(Create(d1), Create(d2), Create(out_layer))

        # FIXED CONNECTION LOGIC
        def get_connections(l1, l2):
            lines = VGroup()
            # Connect all top to all top
            for dot1 in l1[0][0]:
                for dot2 in l2[0][0]:
                    lines.add(Line(dot1.get_center(), dot2.get_center(), stroke_width=0.5, stroke_opacity=0.15))
            # Connect all bottom to all bottom
            for dot1 in l1[0][2]:
                for dot2 in l2[0][2]:
                    lines.add(Line(dot1.get_center(), dot2.get_center(), stroke_width=0.5, stroke_opacity=0.15))
            # Cross-connect for extra lines
            lines.add(Line(l1[0][0][-1].get_center(), l2[0][2][0].get_center(), stroke_width=0.5, stroke_opacity=0.15))
            return lines

        conn1 = get_connections(d1, d2)
        conn2 = get_connections(d2, out_layer)
        self.play(Create(conn1), Create(conn2), run_time=1)

        # --- STAGE 4: DATA FLOW (FIXED SAMPLING) ---
        def animate_dense_flow(conn_group, count=20):
            # FIXED: Set replace=True so we can have more pulses than lines
            path_indices = np.random.choice(len(conn_group), count, replace=True)
            pulses = []
            for idx in path_indices:
                line = conn_group[idx]
                pulse = Dot(radius=0.03, color=YELLOW).move_to(line.get_start())
                # Add a small random delay to pulses sharing the same line
                pulses.append(MoveAlongPath(pulse, line, rate_func=linear, run_time=1.2 + np.random.uniform(-0.2, 0.2)))
            return pulses

        self.play(AnimationGroup(*animate_dense_flow(conn1, 25), lag_ratio=0.05))
        self.play(AnimationGroup(*animate_dense_flow(conn2, 25), lag_ratio=0.05))

        # --- STAGE 5: FINAL SRS MATRIX OUTPUT (4, 1536) ---
        matrix_grid = Rectangle(height=2.0, width=3.0, color=GOLD, fill_opacity=0.2)
        
        # Internal lines to represent the 4 ports
        grid_lines = VGroup()
        for i in range(1, 4):
            y_pos = matrix_grid.get_center()[1] + (i * 0.5 - 1.0)
            grid_lines.add(Line(
                [matrix_grid.get_left()[0], y_pos, 0], 
                [matrix_grid.get_right()[0], y_pos, 0], 
                stroke_width=0.5, stroke_opacity=0.4
            ))
        
        matrix_group = VGroup(matrix_grid, grid_lines).next_to(out_layer, RIGHT, buff=1.0)
        matrix_lbl = Text("Predicted SRS Matrix\n(4 x 1536)", font_size=12).next_to(matrix_group, DOWN)

        self.play(Create(matrix_grid), Create(grid_lines), Write(matrix_lbl))
        self.play(Indicate(matrix_group, color=GOLD))

        denorm_text = Text("Applied Lambda Denormalization", font_size=14, color=BLUE_B).to_edge(DOWN).shift(UP*0.5)
        self.play(Write(denorm_text))

        self.wait(2)