import pygame, math
class Viewer:
    def __init__(self, world_size, mode):
        pygame.init()
        self.scale = 10           # pixels per meter
        self.size = int(world_size * self.scale)
        if mode == "human":
            self.screen = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

    def draw(self, models, types):
        self.screen.fill((30, 30, 30))
        for a, m in models.items():
            color = {"car": (0, 150, 255), "truck": (255, 200, 0), "moto": (255, 50, 50)}[types[a]]
            rect = pygame.Rect(0, 0, m.length*self.scale, m.width*self.scale)
            rect.center = (m.x*self.scale, m.y*self.scale)
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, color, surf.get_rect())
            rot = pygame.transform.rotate(surf, -math.degrees(m.heading))
            self.screen.blit(rot, rot.get_rect(center=rect.center))
        pygame.display.flip()
        self.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
